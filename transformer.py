import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from transformers import MarianTokenizer, MarianMTModel, Trainer, TrainingArguments
import random

# 數據增強函數
def augment_text_char_level(text, p=0.1):
    chars = list(text)
    augmented_chars = []
    for char in chars:
        if random.random() < p:
            operation = random.choice(["delete", "insert", "replace"])
            if operation == "delete":
                continue
            elif operation == "insert":
                augmented_chars.append(random.choice(chars))
            elif operation == "replace":
                augmented_chars.append(random.choice(chars))
        augmented_chars.append(char)
    return "".join(augmented_chars)

# 定義資料集類別
class TranslationDataset(Dataset):
    def __init__(self, tokenizer, texts_zh, texts_tl, max_length=512):
        self.tokenizer = tokenizer
        self.texts_zh = texts_zh
        self.texts_tl = texts_tl
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_zh)

    def __getitem__(self, idx):
        source_text = self.texts_zh[idx]
        target_text = self.texts_tl[idx]

        source_encoded = self.tokenizer(
            source_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        target_encoded = self.tokenizer(
            target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        return {
            'input_ids': source_encoded['input_ids'].squeeze(),
            'attention_mask': source_encoded['attention_mask'].squeeze(),
            'labels': target_encoded['input_ids'].squeeze()
        }

# 測試資料集類別
class TestDataset(Dataset):
    def __init__(self, tokenizer, texts_zh, max_length=512):
        self.tokenizer = tokenizer
        self.texts_zh = texts_zh
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_zh)

    def __getitem__(self, idx):
        source_text = self.texts_zh[idx]

        source_encoded = self.tokenizer(
            source_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        return {
            'input_ids': source_encoded['input_ids'].squeeze(),
            'attention_mask': source_encoded['attention_mask'].squeeze()
        }

# 設置 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 讀取和準備資料
data_folder = "C:\\software\\python\\ml_translate\\translation"
train_zh = pd.read_csv(f"{data_folder}/train-ZH.csv")
train_tl = pd.read_csv(f"{data_folder}/train-TL.csv")

train_zh_txt = train_zh['txt'].str.strip('"')
train_tl_txt = train_tl['txt'].str.strip('"')

# 初始化分詞器
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)

# 隨機選擇八成的訓練數據進行增強
indices_to_augment = random.sample(range(len(train_zh_txt)), int(len(train_zh_txt) * 0.8))

augmented_train_zh = []
augmented_train_tl = []
for idx, (zh, tl) in enumerate(zip(train_zh_txt, train_tl_txt)):
    augmented_train_zh.append(zh)
    augmented_train_tl.append(tl)

    if idx in indices_to_augment:
        augmented_zh = augment_text_char_level(zh)
        augmented_train_zh.append(augmented_zh)
        augmented_train_tl.append(tl)

# 交叉驗證設置
n_splits = 5
kf = KFold(n_splits=n_splits)

for fold, (train_idx, val_idx) in enumerate(kf.split(augmented_train_zh)):
    print(f"Training fold {fold + 1}/{n_splits}")

    train_texts_zh = [augmented_train_zh[i] for i in train_idx]
    train_texts_tl = [augmented_train_tl[i] for i in train_idx]
    val_texts_zh = [augmented_train_zh[i] for i in val_idx]
    val_texts_tl = [augmented_train_tl[i] for i in val_idx]

    train_dataset = TranslationDataset(tokenizer, train_texts_zh, train_texts_tl)
    val_dataset = TranslationDataset(tokenizer, val_texts_zh, val_texts_tl)

    training_args = TrainingArguments(
        output_dir='./results_all_folds2',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=9,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.05,
        logging_dir='./logs_all_folds2',
        logging_steps=10,
    )

    model = MarianMTModel.from_pretrained(model_name).to(device)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()


# 載入測試資料集
test_zh = pd.read_csv(f"{data_folder}/test-ZH-nospace.csv")
test_texts_zh = test_zh['txt'].str.strip('"').tolist()

# 創建測試資料集
test_dataset = TestDataset(tokenizer, test_texts_zh)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# 預測
model.eval()
predictions = []
for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    predictions.extend([tokenizer.decode(t, skip_special_tokens=True) for t in outputs])

# 創建預測結果 DataFrame
predictions_df = pd.DataFrame({
    "id": test_zh['id'],
    "txt": predictions
})

# 將預測結果保存為 CSV 檔案
predictions_df.to_csv(f"{data_folder}/prediction.csv", index=False)
