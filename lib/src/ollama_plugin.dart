import 'dart:async';
import 'dart:convert';

import 'package:dart_monty/dart_monty_bridge.dart';
import 'package:http/http.dart' as http;

/// Plugin that exposes a local [Ollama](https://ollama.com) server to Python
/// code running inside Monty.
///
/// Registers three host functions that mirror the `LlamaMontyExtension` surface
/// so the two plugins are drop-in replacements for each other:
///
/// - `llm_complete(prompt, [model], [system_prompt])` — stateless single-turn
/// - `llm_chat(message, [model], [system_prompt])` — stateful multi-turn
/// - `llm_chat_reset([keep_system_prompt])` — clears conversation history
///
/// **Python usage:**
/// ```python
/// # Stateless — no history retained between calls
/// result = llm_complete("What is 2 + 2?")
/// result = llm_complete("Classify text.", "gemma3", "Reply with one word.")
///
/// # Stateful — history accumulates within the same Monty session
/// r1 = llm_chat("My name is Alice.")
/// r2 = llm_chat("What is my name?")   # model knows "Alice"
/// llm_chat_reset()                    # wipe history
/// ```
///
/// **Dart setup:**
/// ```dart
/// final session = AgentSession(
///   extensions: [OllamaPlugin(defaultModel: 'llama3.2')],
/// );
/// ```
class OllamaPlugin extends MontyExtension {
  /// Creates an [OllamaPlugin].
  ///
  /// [baseUrl] is the Ollama server root (default `http://localhost:11434`).
  /// [defaultModel] is used when the Python caller omits the `model` argument.
  /// [defaultTimeout] caps how long a single completion may take.
  OllamaPlugin({
    String baseUrl = 'http://localhost:11434',
    String defaultModel = 'llama3.2',
    http.Client? client,
    Duration defaultTimeout = const Duration(minutes: 5),
  }) : _baseUrl = baseUrl,
       _defaultModel = defaultModel,
       _client = client ?? http.Client(),
       _defaultTimeout = defaultTimeout;

  final String _baseUrl;
  final String _defaultModel;
  final http.Client _client;
  final Duration _defaultTimeout;

  // Stateful chat history — each entry is {role, content}.
  final List<Map<String, String>> _chatHistory = [];
  String? _chatSystemPrompt;

  @override
  String get namespace => 'llm';

  @override
  String? get systemPromptContext =>
      'Run prompts against a local Ollama server. '
      'Default model: $_defaultModel. '
      'Use llm_complete for one-shot queries, llm_chat for multi-turn '
      'conversations that retain history across calls.';

  @override
  List<HostFunction> get functions => [
    HostFunction(schema: _llmCompleteSchema, handler: _handleComplete),
    HostFunction(schema: _llmChatSchema, handler: _handleChat),
    HostFunction(schema: _llmChatResetSchema, handler: _handleChatReset),
  ];

  /// Child instances share the server config but get isolated chat history.
  @override
  OllamaPlugin createChildInstance({ChildSpawnContext? context}) =>
      OllamaPlugin(
        baseUrl: _baseUrl,
        defaultModel: _defaultModel,
        client: _client,
        defaultTimeout: _defaultTimeout,
      );

  @override
  Future<void> onDispose() async {
    _client.close();
    await super.onDispose();
  }

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  Future<Object?> _handleComplete(Map<String, Object?> args) async {
    final prompt = args.str('prompt');
    final model = args.strOrNull('model') ?? _defaultModel;
    final systemPrompt = args.strOrNull('system_prompt');

    final messages = <Map<String, String>>[
      if (systemPrompt != null && systemPrompt.isNotEmpty)
        {'role': 'system', 'content': systemPrompt},
      {'role': 'user', 'content': prompt},
    ];

    return _chat(model: model, messages: messages);
  }

  Future<Object?> _handleChat(Map<String, Object?> args) async {
    final message = args.str('message');
    final model = args.strOrNull('model') ?? _defaultModel;
    final systemPrompt = args.strOrNull('system_prompt');

    if (systemPrompt != null) {
      _chatSystemPrompt = systemPrompt.isEmpty ? null : systemPrompt;
    }

    _chatHistory.add({'role': 'user', 'content': message});

    final messages = <Map<String, String>>[
      if (_chatSystemPrompt != null)
        {'role': 'system', 'content': _chatSystemPrompt!},
      ..._chatHistory,
    ];

    final reply = await _chat(model: model, messages: messages);

    _chatHistory.add({'role': 'assistant', 'content': reply});

    return reply;
  }

  Future<Object?> _handleChatReset(Map<String, Object?> args) {
    final keepSystem = args['keep_system_prompt'] as bool? ?? true;
    _chatHistory.clear();
    if (!keepSystem) _chatSystemPrompt = null;
    return Future.value();
  }

  // ---------------------------------------------------------------------------
  // HTTP
  // ---------------------------------------------------------------------------

  Future<String> _chat({
    required String model,
    required List<Map<String, String>> messages,
  }) async {
    final uri = Uri.parse('$_baseUrl/api/chat');
    final body = jsonEncode({
      'model': model,
      'messages': messages,
      'stream': false,
    });

    final response = await _client
        .post(
          uri,
          headers: {'content-type': 'application/json'},
          body: body,
        )
        .timeout(_defaultTimeout);

    if (response.statusCode != 200) {
      throw StateError(
        'Ollama returned ${response.statusCode}: ${response.body}',
      );
    }

    final json = jsonDecode(response.body) as Map<String, dynamic>;
    final messageMap = json['message'] as Map<String, dynamic>?;
    final content = messageMap?['content'] as String?;

    if (content == null) {
      throw StateError('Unexpected Ollama response: ${response.body}');
    }

    return content;
  }
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const _llmCompleteSchema = HostFunctionSchema(
  name: 'llm_complete',
  description:
      'Send a prompt to the Ollama server and return the full response '
      'as a string. Stateless — no history is retained between calls.',
  params: [
    HostParam(
      name: 'prompt',
      type: HostParamType.string,
      description: 'The user message to send to the model.',
    ),
    HostParam(
      name: 'model',
      type: HostParamType.string,
      isRequired: false,
      description: 'Ollama model tag to use. Defaults to the plugin default.',
    ),
    HostParam(
      name: 'system_prompt',
      type: HostParamType.string,
      isRequired: false,
      description:
          'Optional system instruction prepended before the prompt.',
    ),
  ],
);

const _llmChatSchema = HostFunctionSchema(
  name: 'llm_chat',
  description:
      'Send a message to the Ollama server and return the reply as a '
      'string. Conversation history is maintained across calls within '
      'the same Monty session.',
  params: [
    HostParam(
      name: 'message',
      type: HostParamType.string,
      description: 'The user message for this turn.',
    ),
    HostParam(
      name: 'model',
      type: HostParamType.string,
      isRequired: false,
      description: 'Ollama model tag to use. Defaults to the plugin default.',
    ),
    HostParam(
      name: 'system_prompt',
      type: HostParamType.string,
      isRequired: false,
      description:
          'System instruction. If provided, replaces the current system '
          'prompt for this and all future turns until changed again.',
    ),
  ],
);

const _llmChatResetSchema = HostFunctionSchema(
  name: 'llm_chat_reset',
  description:
      'Clear the conversation history. Keeps the current system prompt '
      'by default so the next llm_chat starts fresh with the same role.',
  params: [
    HostParam(
      name: 'keep_system_prompt',
      type: HostParamType.boolean,
      isRequired: false,
      defaultValue: true,
      description:
          'Whether to keep the current system prompt. Defaults to True.',
    ),
  ],
);
