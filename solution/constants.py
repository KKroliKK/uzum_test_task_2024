import os


SEED = 42

DATA_PATH = "data"
MODELS_PATH = "models"
FASTTEXT_PATH = "fasttext"
EMBEDDINGS_PATH = "embeddings"

PRODUCTS_PATH = os.path.join(DATA_PATH, "products.parquet")
RETURN_REASONS_PATH = os.path.join(DATA_PATH, "return_reasons.parquet")
RETURNS_PATH = os.path.join(DATA_PATH, "returns.parquet")
REVIEWS_PATH = os.path.join(DATA_PATH, "reviews.parquet")
TEST_PATH = os.path.join(DATA_PATH, "test.parquet")

LID_176_MODEL_PATH = os.path.join(MODELS_PATH, "lid.176.bin")
BEST_BASELINE_MODEL = os.path.join(MODELS_PATH, "best_baseline.cbm")

FASTTEXT_RU_PATH = os.path.join(FASTTEXT_PATH, "cc.ru.300.bin")
FASTTEXT_UZ_PATH = os.path.join(FASTTEXT_PATH, "cc.uz.300.bin")

BAD_REVIEWS_GROUPED_BY_PRODUCT_ID = os.path.join(
    EMBEDDINGS_PATH, "bad_reviews_grouped_by_product_id.pkl"
)
