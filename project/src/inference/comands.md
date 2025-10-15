Для запуска использовались следующие команды:

Для CalibratedRandomForest:

clearml-serving --id d9bc820589c14e0db9c72db9b08ce478 model add --engine custom --endpoint "neres_rf" --preprocess "/home/ahmetov/credit-line/src/inference/rf/preprocess.py" --name "train_rf" --model-id e7144553eac845e296272186096527e9

Для CalibratedLightGBM:

clearml-serving --id d9bc820589c14e0db9c72db9b08ce478 model add --engine custom --endpoint "neres_lgbm" --preprocess "/home/ahmetov/credit-line/src/inference/boosting/preprocess.py" --name "train_lgbm" --model-id 69f38bc48b204011b196973ca4f44850
