#include "targets/qwen3_8/chat_template.h"
#include "artifact/reader.h"
#include "test_checkpoint.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <filesystem>

using namespace xinfer::targets::qwen3_8;
using xinfer::ChatMessage;

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] Line " << __LINE__ << ": " << (msg) << std::endl; \
            return 1; \
        } \
    } while (0)

#define ASSERT_EQ(a, b, msg) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "[FAIL] Line " << __LINE__ << ": " << (msg) \
                      << "\n--- Expected ---\n" << (b) \
                      << "\n--- Actual ---\n" << (a) << std::endl; \
            return 1; \
        } \
    } while (0)

int test_default_template() {
    std::cout << "[Test 1/6] Testing default chat template..." << std::endl;
    QwenChatTemplate tmpl;

    // Single turn user prompt
    std::string expected_single =
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello world!<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n";
    std::string actual_single = tmpl.render("Hello world!");
    ASSERT_EQ(actual_single, expected_single, "Single turn user prompt mismatch");

    // Single turn with system prompt
    std::string expected_sys =
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n"
        "Be concise.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello!<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n";
    std::string actual_sys = tmpl.render("Hello!", "Be concise.");
    ASSERT_EQ(actual_sys, expected_sys, "Single turn with system prompt mismatch");

    std::cout << "  -> Passed: Default template correctly handles single-turn and system prompts." << std::endl;
    return 0;
}

int test_multiturn_and_reasoning() {
    std::cout << "[Test 2/6] Testing multi-turn and assistant reasoning_content..." << std::endl;
    QwenChatTemplate tmpl;

    // Multi-turn without assistant reasoning_content
    std::vector<ChatMessage> msgs = {
        {"user", "Hi", ""},
        {"assistant", "Hello", ""},
        {"user", "How are you?", ""}
    };
    std::string expected_multi =
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hi<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n"
        "</think>\n\n"
        "Hello<|im_end|>\n"
        "<|im_start|>user\n"
        "How are you?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n";
    std::string actual_multi = tmpl.render(msgs);
    ASSERT_EQ(actual_multi, expected_multi, "Multi-turn formatting mismatch");

    // Multi-turn with assistant reasoning_content
    std::vector<ChatMessage> msgs_reasoning = {
        {"user", "Hi", ""},
        {"assistant", "Hello", "Let me think"},
        {"user", "Next", ""}
    };
    std::string expected_reasoning =
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hi<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "Let me think\n"
        "</think>\n\n"
        "Hello<|im_end|>\n"
        "<|im_start|>user\n"
        "Next<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n";
    std::string actual_reasoning = tmpl.render(msgs_reasoning);
    ASSERT_EQ(actual_reasoning, expected_reasoning, "Reasoning content formatting mismatch");

    std::cout << "  -> Passed: Multi-turn and reasoning_content properly formatted." << std::endl;
    return 0;
}

int test_template_options() {
    std::cout << "[Test 3/6] Testing template options (enable_thinking, reasoning_effort, add_generation_prompt)..." << std::endl;
    QwenChatTemplate tmpl;
    std::vector<ChatMessage> msgs = {{"user", "Hello!", ""}};

    // 1. enable_thinking = false
    ChatTemplateOptions opt_no_thinking;
    opt_no_thinking.enable_thinking = false;
    std::string expected_no_thinking =
        "<|im_start|>user\n"
        "Hello!<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n"
        "</think>\n\n";
    std::string actual_no_thinking = tmpl.render(msgs, opt_no_thinking);
    ASSERT_EQ(actual_no_thinking, expected_no_thinking, "enable_thinking=false mismatch");

    // 2. reasoning_effort = "low"
    ChatTemplateOptions opt_low;
    opt_low.reasoning_effort = "low";
    std::string expected_low =
        "<|im_start|>system\n"
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello!<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n";
    std::string actual_low = tmpl.render(msgs, opt_low);
    ASSERT_EQ(actual_low, expected_low, "reasoning_effort='low' mismatch");

    // 3. reasoning_effort = "medium" (instructions are empty)
    ChatTemplateOptions opt_med;
    opt_med.reasoning_effort = "medium";
    std::string expected_med =
        "<|im_start|>user\n"
        "Hello!<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n";
    std::string actual_med = tmpl.render(msgs, opt_med);
    ASSERT_EQ(actual_med, expected_med, "reasoning_effort='medium' mismatch");

    // 4. add_generation_prompt = false
    ChatTemplateOptions opt_no_gen;
    opt_no_gen.add_generation_prompt = false;
    std::string expected_no_gen =
        "<|im_start|>system\n"
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello!<|im_end|>\n";
    std::string actual_no_gen = tmpl.render(msgs, opt_no_gen);
    ASSERT_EQ(actual_no_gen, expected_no_gen, "add_generation_prompt=false mismatch");

    std::cout << "  -> Passed: Template options faithfully rendered." << std::endl;
    return 0;
}

int test_load_from_file_and_artifact() {
    std::cout << "[Test 4/6] Testing loading real chat_template.jinja from checkpoint and artifact..." << std::endl;

    // 1. From checkpoint file
    std::filesystem::path checkpoint_dir = xinfer::test::get_checkpoint_dir();
    std::filesystem::path jinja_path = checkpoint_dir / "chat_template.jinja";
    if (std::filesystem::exists(jinja_path)) {
        QwenChatTemplate tmpl;
        std::string err;
        bool ok = tmpl.load_from_file(jinja_path.string(), &err);
        ASSERT_TRUE(ok, ("Failed to load chat_template.jinja from file: " + err).c_str());
        ASSERT_TRUE(tmpl.is_loaded(), "is_loaded() must be true");
        ASSERT_EQ(tmpl.default_reasoning_effort(), "xhigh", "Default reasoning effort must be xhigh");

        std::string actual = tmpl.render("Hello world!");
        std::string expected =
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
            "<|im_start|>user\n"
            "Hello world!<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n";
        ASSERT_EQ(actual, expected, "Render from loaded jinja file mismatch");
        std::cout << "  -> Passed: Loaded and verified real chat_template.jinja from checkpoint." << std::endl;
    } else {
        std::cout << "  -> Note: Checkpoint chat_template.jinja not accessible ("
                  << jinja_path.string() << "), skipping file load test." << std::endl;
    }

    // 2. From .xinfer container artifact
    std::string artifact_path = "out/qwen3_8_27b.xinfer";
    if (std::filesystem::exists(artifact_path)) {
        xinfer::artifact::ArtifactReader reader;
        std::string open_err;
        if (reader.open(artifact_path, &open_err)) {
            ASSERT_TRUE(reader.has_section("chat_template.jinja"), "Artifact must contain section 'chat_template.jinja'");
            std::vector<uint8_t> data;
            std::string read_err;
            bool read_ok = reader.read_section("chat_template.jinja", data, &read_err);
            ASSERT_TRUE(read_ok, ("Failed to read chat_template.jinja section from artifact: " + read_err).c_str());
            ASSERT_TRUE(!data.empty(), "chat_template.jinja payload must not be empty");

            QwenChatTemplate artifact_tmpl;
            std::string load_err;
            bool load_ok = artifact_tmpl.load_from_buffer(data.data(), data.size(), &load_err);
            ASSERT_TRUE(load_ok, ("Failed to parse chat template from artifact buffer: " + load_err).c_str());
            ASSERT_TRUE(artifact_tmpl.is_loaded(), "is_loaded() must be true after loading from artifact");

            std::string actual = artifact_tmpl.render("Hello world!");
            std::string expected =
                "<|im_start|>system\n"
                "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n"
                "<|im_start|>user\n"
                "Hello world!<|im_end|>\n"
                "<|im_start|>assistant\n"
                "<think>\n";
            ASSERT_EQ(actual, expected, "Render from artifact chat template mismatch");
            std::cout << "  -> Passed: Read and verified real chat_template.jinja from .xinfer container artifact." << std::endl;
        }
    } else {
        std::cout << "  -> Note: out/qwen3_8_27b.xinfer not found, skipping artifact read test." << std::endl;
    }

    return 0;
}

int test_error_handling() {
    std::cout << "[Test 5/6] Testing error handling (fail-loudly)..." << std::endl;
    QwenChatTemplate tmpl;

    // 1. Empty messages
    {
        std::vector<ChatMessage> empty_msgs;
        std::string err;
        std::string res = tmpl.render(empty_msgs, {}, &err);
        ASSERT_TRUE(res.empty(), "Empty messages must return empty string");
        ASSERT_TRUE(err.find("No messages provided") != std::string::npos, "Must report 'No messages provided'");
    }

    // 2. System message not at index 0
    {
        std::vector<ChatMessage> bad_sys = {
            {"user", "Hello", ""},
            {"system", "You are an assistant.", ""}
        };
        std::string err;
        std::string res = tmpl.render(bad_sys, {}, &err);
        ASSERT_TRUE(res.empty(), "System message not at beginning must return empty string");
        ASSERT_TRUE(err.find("System message must be at the beginning") != std::string::npos,
                    "Must report 'System message must be at the beginning'");
    }

    // 3. Invalid reasoning effort
    {
        std::vector<ChatMessage> msgs = {{"user", "Hello", ""}};
        ChatTemplateOptions bad_opt;
        bad_opt.reasoning_effort = "maximum";
        std::string err;
        std::string res = tmpl.render(msgs, bad_opt, &err);
        ASSERT_TRUE(res.empty(), "Invalid reasoning effort must return empty string");
        ASSERT_TRUE(err.find("Unexpected reasoning effort") != std::string::npos,
                    "Must report 'Unexpected reasoning effort'");
    }

    // 4. Invalid message role
    {
        std::vector<ChatMessage> bad_role = {
            {"user", "Hello", ""},
            {"alien", "Greetings", ""}
        };
        std::string err;
        std::string res = tmpl.render(bad_role, {}, &err);
        ASSERT_TRUE(res.empty(), "Unknown role must return empty string");
        ASSERT_TRUE(err.find("Unexpected message role") != std::string::npos,
                    "Must report 'Unexpected message role'");
    }

    // 5. No user query in messages
    {
        std::vector<ChatMessage> no_user = {{"system", "System instruction.", ""}};
        std::string err;
        std::string res = tmpl.render(no_user, {}, &err);
        ASSERT_TRUE(res.empty(), "No user query must return empty string");
        ASSERT_TRUE(err.find("No user query found in messages") != std::string::npos,
                    "Must report 'No user query found in messages'");
    }

    std::cout << "  -> Passed: All error conditions caught and reported." << std::endl;
    return 0;
}

int test_template_parsing_and_validation() {
    std::cout << "[Test 6/6] Testing template parsing, fail-loudly validation, and custom template driving..." << std::endl;
    QwenChatTemplate tmpl;

    // 1. Empty template loading must fail loudly
    {
        std::string err;
        bool ok = tmpl.load_from_string("", &err);
        ASSERT_TRUE(!ok, "Loading empty template string must return false");
        ASSERT_TRUE(!tmpl.is_loaded(), "is_loaded() must be false after failed load");
        ASSERT_TRUE(err.find("Empty chat template") != std::string::npos, "Must report 'Empty chat template'");
    }

    // 2. Non-ChatML template (e.g. Llama-3 format) must fail structural validation loudly
    {
        std::string llama_template =
            "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>' -}}\n"
            "{{- message['content'] -}}\n"
            "{{- '<|eot_id|>' -}}";
        std::string err;
        bool ok = tmpl.load_from_string(llama_template, &err);
        ASSERT_TRUE(!ok, "Loading non-ChatML template must return false");
        ASSERT_TRUE(!tmpl.is_loaded(), "is_loaded() must be false after failed load");
        ASSERT_TRUE(err.find("Template structural validation failed") != std::string::npos,
                    "Must report structural validation failure");
    }

    // 3. Template missing default reasoning effort pattern must fail loudly
    {
        std::string missing_default_tmpl =
            "<|im_start|>system\n{{ content }}<|im_end|>\n"
            "<think>{{ reasoning }}</think>\n"
            "{% if add_generation_prompt %}{% endif %}\n"
            "{% if enable_thinking %}{% endif %}\n"
            "{% if multi_step_tool %}<tool_response></tool_response>{% endif %}\n"
            "{% for message in messages %}{{ system }} {{ user }} {{ assistant }}{% endfor %}\n";
        std::string err;
        bool ok = tmpl.load_from_string(missing_default_tmpl, &err);
        ASSERT_TRUE(!ok, "Template missing default reasoning effort must return false");
        ASSERT_TRUE(!tmpl.is_loaded(), "is_loaded() must be false");
        ASSERT_TRUE(err.find("default reasoning effort") != std::string::npos,
                    "Must report failure to extract default reasoning effort");
    }

    // 4. Template missing reasoning instructions for 'xhigh' must fail loudly
    {
        std::string missing_xhigh_tmpl =
            "<|im_start|>system\n{{ content }}<|im_end|>\n"
            "<think>{{ reasoning }}</think>\n"
            "{% if add_generation_prompt %}{% endif %}\n"
            "{% if enable_thinking %}{% endif %}\n"
            "{% if multi_step_tool %}<tool_response></tool_response>{% endif %}\n"
            "{% for message in messages %}{{ system }} {{ user }} {{ assistant }}{% endfor %}\n"
            "{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
            "{% if resolved_reasoning_effort == 'low' %}\n"
            "    {% set reasoning_instructions = 'Brief thinking.' %}\n"
            "{% endif %}\n";
        std::string err;
        bool ok = tmpl.load_from_string(missing_xhigh_tmpl, &err);
        ASSERT_TRUE(!ok, "Template missing xhigh instructions must return false");
        ASSERT_TRUE(!tmpl.is_loaded(), "is_loaded() must be false");
        ASSERT_TRUE(err.find("effort 'xhigh'") != std::string::npos,
                    "Must report failure to extract reasoning instructions for xhigh");
    }

    // 5. Valid template with custom instructions and default effort driving the renderer
    {
        std::string custom_template =
            "<|im_start|>system\n"
            "{% set reasoning_instructions = '' %}\n"
            "{% if enable_thinking is undefined or enable_thinking is true %}\n"
            "    {% set resolved_reasoning_effort = reasoning_effort | default( 'low' ) %}\n"
            "    {% if resolved_reasoning_effort == 'xhigh' %}\n"
            "        {% set reasoning_instructions = 'CUSTOM_XHIGH_INSTRUCTION_ALPHA' %}\n"
            "    {% elif resolved_reasoning_effort == 'low' %}\n"
            "        {% set reasoning_instructions = 'CUSTOM_LOW_INSTRUCTION_BETA' %}\n"
            "    {% endif %}\n"
            "{% endif %}\n"
            "<think>{{ reasoning }}</think>\n"
            "{% if add_generation_prompt %}{% endif %}\n"
            "{% if multi_step_tool %}<tool_response></tool_response>{% endif %}\n"
            "{% for message in messages %}<|im_start|>{{ message.role }}{{ system }} {{ user }} {{ assistant }}<|im_end|>{% endfor %}\n";

        std::string err;
        bool ok = tmpl.load_from_string(custom_template, &err);
        ASSERT_TRUE(ok, ("Failed to load valid custom template: " + err).c_str());
        ASSERT_TRUE(tmpl.is_loaded(), "is_loaded() must be true after loading valid custom template");
        ASSERT_EQ(tmpl.default_reasoning_effort(), "low", "Default reasoning effort must be extracted as 'low'");
        ASSERT_EQ(tmpl.get_reasoning_instructions("xhigh"), "CUSTOM_XHIGH_INSTRUCTION_ALPHA",
                  "Custom xhigh instructions must match template");
        ASSERT_EQ(tmpl.get_reasoning_instructions("low"), "CUSTOM_LOW_INSTRUCTION_BETA",
                  "Custom low instructions must match template");

        // Verify that render() uses the template-driven default reasoning effort ("low")
        std::string rendered_default = tmpl.render("Test message");
        ASSERT_TRUE(rendered_default.find("CUSTOM_LOW_INSTRUCTION_BETA") != std::string::npos,
                    "render() must use template-driven 'low' reasoning instructions by default");
        ASSERT_TRUE(rendered_default.find("Reasoning effort is set to") == std::string::npos,
                    "render() must NOT use hardcoded fallback English text when template is loaded");

        // Verify that render() with reasoning_effort='xhigh' uses the template-driven 'xhigh' instructions
        ChatTemplateOptions opt_xhigh;
        opt_xhigh.reasoning_effort = "xhigh";
        std::string rendered_xhigh = tmpl.render("Test message", "", opt_xhigh);
        ASSERT_TRUE(rendered_xhigh.find("CUSTOM_XHIGH_INSTRUCTION_ALPHA") != std::string::npos,
                    "render() must use template-driven 'xhigh' reasoning instructions");
    }

    std::cout << "  -> Passed: Template parsing, structural fail-loudly validation, and dynamic rendering verified." << std::endl;
    return 0;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << " xinfer Real Chat Template Verification & Test" << std::endl;
    std::cout << "==========================================================" << std::endl;

    if (test_default_template() != 0) return 1;
    if (test_multiturn_and_reasoning() != 0) return 1;
    if (test_template_options() != 0) return 1;
    if (test_load_from_file_and_artifact() != 0) return 1;
    if (test_error_handling() != 0) return 1;
    if (test_template_parsing_and_validation() != 0) return 1;

    std::cout << "==========================================================" << std::endl;
    std::cout << " ALL CHAT TEMPLATE TESTS PASSED SUCCESSFULLY!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
