from __future__ import annotations

import hashlib
import uuid

import pandas as pd
from data.config import QUOTES_OUTPUT, RAW_DATA_DIR


def resolve_vehicles() -> pd.DataFrame:
    quotes_df = pd.read_parquet(QUOTES_OUTPUT)

    vehicles: list[dict] = []

    for _, row in quotes_df.iterrows():
        vin = _generate_vin(row["quote_id"], row["vehicle_power"], row["vehicle_msrp"])
        vehicles.append({
            "vin": vin,
            "quote_id": str(row["quote_id"]),
            "vehicle_id": str(uuid.uuid4()),
            "year": _decode_vin_year(vin),
            "make": _decode_vin_make(vin),
            "model": _decode_vin_model(vin),
            "body_type": "sedan",
            "msrp": float(row["vehicle_msrp"]),
            "horsepower": float(row["vehicle_power"]),
            "adas_score": float(row["vehicle_adas_score"]),
        })

    vehicles_df = pd.DataFrame(vehicles)

    entities_dir = RAW_DATA_DIR / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    output_path = entities_dir / "vehicles.parquet"
    vehicles_df.to_parquet(output_path, index=False)

    assert vehicles_df["vin"].isnull().sum() == 0, "No null VINs allowed"
    print(f"Resolved {len(vehicles_df)} unique vehicles to {output_path}")
    return vehicles_df


def _generate_vin(quote_id: str, power: float, msrp: float) -> str:
    seed = f"{quote_id}:{power}:{msrp}"
    hash_obj = hashlib.sha256(seed.encode())
    hash_hex = hash_obj.hexdigest()
    make_code = hash_hex[0:2]
    model_code = hash_hex[2:4]
    checksum = hash_hex[4:6]
    return f"1{make_code}V{model_code}D2{checksum}000001"


def _decode_vin_year(vin: str) -> int:
    year_code = vin[9] if len(vin) > 9 else "0"
    year_map = {
        "0": 2020, "1": 2021, "2": 2022, "3": 2023, "4": 2024,
        "5": 2025, "6": 2026, "7": 2019, "8": 2018, "9": 2017,
        "A": 2010, "B": 2011, "C": 2012, "D": 2013, "E": 2014,
        "F": 2015, "G": 2016, "H": 2017, "J": 2018, "K": 2019,
        "L": 2020, "M": 2021, "N": 2022, "P": 2023, "R": 2024,
        "S": 2025, "T": 2026, "V": 2027, "W": 2028, "X": 2029,
        "Y": 2030, "Z": 2031,
    }
    return year_map.get(year_code.upper(), 2023)


def _decode_vin_make(vin: str) -> str:
    make_code = vin[1:3] if len(vin) > 2 else "00"
    make_map = {
        "0": "Toyota", "1": "Honda", "2": "Ford", "3": "BMW", "4": "Audi",
        "5": "Mercedes", "6": "Volkswagen", "7": "Nissan", "8": "Hyundai", "9": "Kia",
        "A": "Mazda", "B": "Subaru", "C": "Volvo", "D": "Jaguar", "E": "Lexus",
        "F": "Tesla", "G": "Chevrolet", "H": "GMC", "I": "Cadillac", "J": "Buick",
    }
    first_char = make_code[0].upper() if make_code else "0"
    return make_map.get(first_char, "Toyota")


def _decode_vin_model(vin: str) -> str:
    model_code = vin[3:5] if len(vin) > 4 else "00"
    model_map = {
        "0": "Civic", "1": "Accord", "2": "Corolla", "3": "Camry", "4": "Model 3",
        "5": "Model Y", "6": "Mustang", "7": "Escape", "8": "F-150", "9": "Ram 1500",
        "A": "A4", "B": "3 Series", "C": "C-Class", "D": "Q5", "E": "Q7",
    }
    first_char = model_code[0].upper() if model_code else "0"
    return model_map.get(first_char, "Civic")
