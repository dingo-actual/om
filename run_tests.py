import orjson

from src.tests import test_arc_transformer, test_run_dataloaders, test_model, test_model_training


if __name__ == "__main__":
    test_arc_transformer()
    
    test_model()
    
    result = test_run_dataloaders(
        config_path="C:\\Users\\photo\\projects\\bak\\om\\config\\data.json",
        dataloader_kwargs={
            "num_workers": 16,
            "shuffle": False,
            "drop_last": True,
        }
    )
    
    with open("token_counts.json", "wb") as fp:
        fp.write(orjson.dumps(result))
    
    test_model_training()