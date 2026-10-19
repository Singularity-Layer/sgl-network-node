use crate::{config, orchestrator::OrchestratorClient};
use std::path::Path;

#[derive(Debug, PartialEq)]
enum PriceUpdate {
    Tokens { input_per_m: f64, output_per_m: f64 },
    Audio { per_second: f64 },
}

fn price_update(
    input: Option<f64>,
    output: Option<f64>,
    audio_second: Option<f64>,
) -> Result<PriceUpdate, String> {
    match (input, output, audio_second) {
        (Some(input_per_m), Some(output_per_m), None)
            if input_per_m.is_finite()
                && input_per_m >= 0.0
                && output_per_m.is_finite()
                && output_per_m >= 0.0 =>
        {
            Ok(PriceUpdate::Tokens {
                input_per_m,
                output_per_m,
            })
        }
        (None, None, Some(per_second)) if per_second.is_finite() && per_second > 0.0 => {
            Ok(PriceUpdate::Audio { per_second })
        }
        _ => Err("Choose either --input with --output, or --audio-second by itself.".into()),
    }
}

/// Show effective pricing without assuming every serving model is token-priced.
pub async fn show_prices(config_dir: &Path, orchestrator_url: &str) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    let data = client.get_prices(&cfg.node_id).await?;
    let prices = data
        .get("prices")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();
    if prices.is_empty() {
        println!("This node isn't serving any models yet.");
        return Ok(());
    }
    println!("Per-model pricing:\n");
    for price in prices {
        let model = price.get("model").and_then(|m| m.as_str()).unwrap_or("?");
        let custom = price.get("custom").is_some_and(|value| !value.is_null());
        if price.get("price_unit").and_then(|unit| unit.as_str()) == Some("audio_second") {
            let effective = price
                .get("effective")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let reference = price
                .get("reference")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let floor = price.get("floor").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let ceiling = price.get("ceiling").and_then(|v| v.as_f64()).unwrap_or(0.0);
            println!(
                "  {model:<20} ${effective:.6}/audio second [{}]   (suggested ${reference:.6}; band ${floor:.6}–${ceiling:.6}; you keep ${:.6})",
                if custom { "custom" } else { "suggested" }, effective * 0.8,
            );
            continue;
        }
        let field = |container: Option<&serde_json::Value>, key: &str| {
            container
                .and_then(|value| value.get(key))
                .and_then(|n| n.as_f64())
                .unwrap_or(0.0)
        };
        let effective = price.get("effective");
        let reference = price.get("reference");
        println!(
            "  {model:<20} in ${:.6} / out ${:.6} per 1M [{}]   (suggested in ${:.6} / out ${:.6})",
            field(effective, "inputPerM"),
            field(effective, "outputPerM"),
            if custom { "custom" } else { "suggested" },
            field(reference, "inputPerM"),
            field(reference, "outputPerM"),
        );
    }
    Ok(())
}

pub async fn set_price(
    config_dir: &Path,
    orchestrator_url: &str,
    model: &str,
    input: Option<f64>,
    output: Option<f64>,
    audio_second: Option<f64>,
) -> Result<(), String> {
    let update = price_update(input, output, audio_second)?;
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    match update {
        PriceUpdate::Tokens {
            input_per_m,
            output_per_m,
        } => {
            client
                .set_price(&cfg.node_id, model, input_per_m, output_per_m)
                .await?;
            println!("✅ Price set for {model}: in ${input_per_m}/1M · out ${output_per_m}/1M. You keep 80% of what you charge.");
        }
        PriceUpdate::Audio { per_second } => {
            client
                .set_audio_price(&cfg.node_id, model, per_second)
                .await?;
            println!("✅ Price set for {model}: ${per_second:.6}/audio second. You keep 80% of what you charge.");
        }
    }
    Ok(())
}

pub async fn reset_price(
    config_dir: &Path,
    orchestrator_url: &str,
    model: &str,
) -> Result<(), String> {
    let cfg = config::load_config(config_dir)?;
    let client = OrchestratorClient::new(orchestrator_url, Some(cfg.auth_token.clone()));
    client.reset_price(&cfg.node_id, model).await?;
    println!("✅ {model} reset to the platform suggested price.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_exactly_one_pricing_unit() {
        assert_eq!(
            price_update(Some(1.0), Some(2.0), None).unwrap(),
            PriceUpdate::Tokens {
                input_per_m: 1.0,
                output_per_m: 2.0
            }
        );
        assert_eq!(
            price_update(None, None, Some(0.0001)).unwrap(),
            PriceUpdate::Audio { per_second: 0.0001 }
        );
        for invalid in [
            price_update(None, None, None),
            price_update(Some(1.0), None, None),
            price_update(None, Some(2.0), None),
            price_update(Some(1.0), Some(2.0), Some(0.0001)),
        ] {
            assert!(invalid.is_err());
        }
    }
}
