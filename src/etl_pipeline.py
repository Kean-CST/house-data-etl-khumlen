"""
House Sale Data ETL Pipeline
============================
Implement the three functions below to complete the ETL pipeline.

Steps:
  1. EXTRACT  – load the CSV into a PySpark DataFrame
  2. TRANSFORM – split the data by neighborhood and save each as a separate CSV
  3. LOAD      – insert each neighborhood DataFrame into its own PostgreSQL table
"""
from __future__ import annotations

import csv  # noqa: F401
import os  # noqa: F401
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401
from pyspark.sql import DataFrame, SparkSession  # noqa: F401
from pyspark.sql import functions as F  # noqa: F401

# ── Predefined constants (do not modify) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

NEIGHBORHOODS = [
    "Downtown", "Green Valley", "Hillcrest", "Lakeside", "Maple Heights",
    "Oakwood", "Old Town", "Riverside", "Suburban Park", "University District",
]

OUTPUT_DIR   = ROOT / "output" / "by_neighborhood"
OUTPUT_FILES = {hood: OUTPUT_DIR / f"{hood.replace(' ', '_').lower()}.csv" for hood in NEIGHBORHOODS}

PG_TABLES = {hood: f"public.{hood.replace(' ', '_').lower()}" for hood in NEIGHBORHOODS}

PG_COLUMN_SCHEMA = (
    "house_id TEXT, neighborhood TEXT, price INTEGER, square_feet INTEGER, "
    "num_bedrooms INTEGER, num_bathrooms INTEGER, house_age INTEGER, "
    "garage_spaces INTEGER, lot_size_acres NUMERIC(6,2), has_pool BOOLEAN, "
    "recently_renovated BOOLEAN, energy_rating TEXT, location_score INTEGER, "
    "school_rating INTEGER, crime_rate INTEGER, "
    "distance_downtown_miles NUMERIC(6,2), sale_date DATE, days_on_market INTEGER"
)


def extract(spark: SparkSession, csv_path: str) -> DataFrame:
    """Load the CSV dataset into a PySpark DataFrame with correct data types."""
    df = spark.read.option("header", True).csv(csv_path)

    int_columns = [
        "price",
        "square_feet",
        "num_bedrooms",
        "num_bathrooms",
        "house_age",
        "garage_spaces",
        "location_score",
        "school_rating",
        "crime_rate",
        "days_on_market",
        "buyer_budget",
        "buyer_family_size",
    ]
    float_columns = ["lot_size_acres", "distance_downtown_miles"]
    bool_columns = ["has_pool", "recently_renovated", "has_children", "first_time_buyer"]

    for column in int_columns:
        df = df.withColumn(column, F.col(column).cast("int"))

    for column in float_columns:
        df = df.withColumn(column, F.col(column).cast("double"))

    for column in bool_columns:
        df = df.withColumn(column, (F.upper(F.trim(F.col(column))) == F.lit("TRUE")))

    df = df.withColumn("sale_date", F.to_date(F.col("sale_date"), "M/d/yy"))
    return df


def transform(df: DataFrame) -> dict[str, DataFrame]:
    """Split the data by neighborhood and save each as a separate CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitions: dict[str, DataFrame] = {}

    def _serialize(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            # Match expected CSV formatting (e.g. 0 instead of 0.0)
            if value.is_integer():
                return str(int(value))
            return str(value)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    for hood in NEIGHBORHOODS:
        hood_df = df.filter(F.col("neighborhood") == hood).orderBy("house_id")
        partitions[hood] = hood_df

        with OUTPUT_FILES[hood].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(hood_df.columns)
            for row in hood_df.collect():
                writer.writerow([_serialize(value) for value in row])

    return partitions


def load(partitions: dict[str, DataFrame], jdbc_url: str, pg_props: dict) -> None:
    """Insert each neighborhood dataset into its own PostgreSQL table."""
    db_columns = [
        "house_id",
        "neighborhood",
        "price",
        "square_feet",
        "num_bedrooms",
        "num_bathrooms",
        "house_age",
        "garage_spaces",
        "lot_size_acres",
        "has_pool",
        "recently_renovated",
        "energy_rating",
        "location_score",
        "school_rating",
        "crime_rate",
        "distance_downtown_miles",
        "sale_date",
        "days_on_market",
    ]

    for hood, hood_df in partitions.items():
        (
            hood_df.select(*db_columns)
            .write
            .mode("overwrite")
            .jdbc(
                url=jdbc_url,
                table=PG_TABLES[hood],
                properties={**pg_props, "createTableColumnTypes": PG_COLUMN_SCHEMA},
            )
        )


# ── Main (do not modify) ───────────────────────────────────────────────────────
def main() -> None:
    load_dotenv(ROOT / ".env")

    jdbc_url = (
        f"jdbc:postgresql://{os.getenv('PG_HOST', 'localhost')}:"
        f"{os.getenv('PG_PORT', '5432')}/{os.environ['PG_DATABASE']}"
    )
    pg_props = {
        "user":     os.environ["PG_USER"],
        "password": os.getenv("PG_PASSWORD", ""),
        "driver":   "org.postgresql.Driver",
    }
    csv_path = str(ROOT / os.getenv("DATASET_DIR", "dataset") / os.getenv("DATASET_FILE", "historical_purchases.csv"))

    spark = (
        SparkSession.builder.appName("HouseSaleETL")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df         = extract(spark, csv_path)
    partitions = transform(df)
    load(partitions, jdbc_url, pg_props)

    spark.stop()


if __name__ == "__main__":
    main()
