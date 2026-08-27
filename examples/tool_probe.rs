use sgl_node::inprocess::{InProcessConfig, InProcessEngine};
use sgl_node::inference::ChatMessage;

#[tokio::main]
async fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let e = InProcessEngine::start(InProcessConfig {
        model_path: path.clone().into(),
        model_name: "llama-3.2-3b".into(),
        n_ctx: 8192, n_gpu_layers: 999, max_slots: 1, per_slot_ctx: 8192,
        mmproj_path: None, image_max_tokens: None,
    }).await.expect("engine start");
    println!("tool_format detected: {:?}", e.tool_format());
    let tools = serde_json::json!([{
        "type":"function",
        "function":{"name":"get_weather","description":"Get the weather for a city",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}
    }]);
    let msgs = vec![ChatMessage{ role:"user".into(), content:"What is the weather in Paris? Use the tool.".into() }];
    let out = e.chat_completion(&msgs, Vec::new(), Some(tools), vec!["get_weather".into()], 80, 0.0)
        .await.expect("completion");
    println!("finish_reason : {:?}", out.finish_reason);
    println!("tool_calls    : {}", serde_json::to_string(&out.tool_calls).unwrap());
    println!("content       : {:?}", out.content);
    println!("tokens        : {} prompt / {} completion", out.prompt_tokens, out.completion_tokens);
}
