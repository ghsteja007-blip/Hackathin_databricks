from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import pandas as pd


SCHEMA_DEFAULT = "databricks_virtue_foundation_dataset_dais_2026"


@dataclass(frozen=True)
class Datasets:
    facilities: pd.DataFrame
    pincodes: pd.DataFrame
    nfhs: pd.DataFrame


def _safe_identifier_part(part: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
        raise ValueError(f"Unsafe SQL identifier: {part!r}")
    return f"`{part}`"


def _qualified_table(default_table: str, env_name: str) -> str:
    table_value = os.getenv(env_name, default_table).strip()
    schema = os.getenv("DATABRICKS_SCHEMA", SCHEMA_DEFAULT).strip()
    catalog = os.getenv("DATABRICKS_CATALOG", "").strip()

    if "." in table_value:
        parts = table_value.split(".")
    else:
        parts = [schema, table_value] if not catalog else [catalog, schema, table_value]

    return ".".join(_safe_identifier_part(part) for part in parts if part)


def _server_hostname() -> str | None:
    explicit = os.getenv("DATABRICKS_SERVER_HOSTNAME")
    if explicit:
        return explicit.replace("https://", "").rstrip("/")

    host = os.getenv("DATABRICKS_HOST")
    if not host:
        return None
    return urlparse(host).netloc or host.replace("https://", "").rstrip("/")


def _http_path() -> str | None:
    if os.getenv("DATABRICKS_HTTP_PATH"):
        return os.getenv("DATABRICKS_HTTP_PATH")

    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID") or os.getenv("WAREHOUSE_ID")
    if warehouse_id:
        return f"/sql/1.0/warehouses/{warehouse_id}"
    return None


def _databricks_configured() -> bool:
    return bool(_server_hostname() and _http_path())


def _databricks_connection():
    from databricks import sql

    server_hostname = _server_hostname()
    http_path = _http_path()
    token = os.getenv("DATABRICKS_TOKEN")

    if token:
        return sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
            user_agent_entry="referral-copilot",
        )

    from databricks.sdk.core import Config, oauth_service_principal

    def credential_provider():
        config = Config(
            host=f"https://{server_hostname}",
            client_id=os.getenv("DATABRICKS_CLIENT_ID"),
            client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
        )
        return oauth_service_principal(config)

    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        credentials_provider=credential_provider,
        user_agent_entry="referral-copilot",
    )


def _read_sql(query: str, parameters: Iterable | None = None) -> pd.DataFrame:
    with _databricks_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters=list(parameters or []))
            rows = cursor.fetchall()
            columns = [field[0] for field in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _read_databricks_table(default_table: str, env_name: str, columns: list[str], limit_env: str) -> pd.DataFrame:
    table_name = _qualified_table(default_table, env_name)
    limit = int(os.getenv(limit_env, os.getenv("TABLE_ROW_LIMIT", "50000")))
    selected_columns = ", ".join(_safe_identifier_part(col) for col in columns)
    query = f"SELECT {selected_columns} FROM {table_name} LIMIT ?"
    return _read_sql(query, [limit])


def _local_csv_path(filename: str) -> Path | None:
    candidates: list[Path] = []

    if os.getenv("LOCAL_DATA_DIR"):
        candidates.append(Path(os.getenv("LOCAL_DATA_DIR", "")) / filename)

    candidates.extend(
        [
            Path.cwd() / "data" / filename,
            Path.cwd() / filename,
            Path.home() / "Downloads" / filename,
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return None


def _read_local_csv(filename: str) -> pd.DataFrame:
    path = _local_csv_path(filename)
    if not path:
        raise FileNotFoundError(
            f"Could not find {filename}. Set LOCAL_DATA_DIR or configure Databricks SQL resources."
        )
    return pd.read_csv(path, low_memory=False)


def _load_from_databricks() -> tuple[Datasets, list[str]]:
    facility_columns = [
        "unique_id",
        "name",
        "facilityTypeId",
        "operatorTypeId",
        "description",
        "phone_numbers",
        "officialPhone",
        "email",
        "websites",
        "officialWebsite",
        "address_line1",
        "address_line2",
        "address_line3",
        "address_city",
        "address_stateOrRegion",
        "address_zipOrPostcode",
        "address_country",
        "latitude",
        "longitude",
        "specialties",
        "procedure",
        "equipment",
        "capability",
        "capacity",
        "numberDoctors",
        "distinct_social_media_presence_count",
        "source_urls",
    ]
    pincode_columns = [
        "circlename",
        "regionname",
        "divisionname",
        "officename",
        "pincode",
        "officetype",
        "delivery",
        "district",
        "statename",
        "latitude",
        "longitude",
    ]
    nfhs_columns = [
        "district_name",
        "state_ut",
        "hh_improved_water_pct",
        "hh_use_improved_sanitation_pct",
        "households_using_clean_fuel_for_cooking_pct",
        "hh_member_covered_health_insurance_pct",
        "institutional_birth_5y_pct",
        "child_12_23m_fully_vaccinated_based_on_information_from_vax_pct",
        "child_u5_who_are_stunted_height_for_age_18_pct",
        "child_6_59m_who_are_anaemic_lt_11_0_g_dl_22_pct",
        "all_w15_49_who_are_anaemic_pct",
    ]

    return (
        Datasets(
            facilities=_read_databricks_table("facilities", "FACILITIES_TABLE", facility_columns, "FACILITY_ROW_LIMIT"),
            pincodes=_read_databricks_table(
                "india_post_pincode_directory",
                "PINCODE_TABLE",
                pincode_columns,
                "PINCODE_ROW_LIMIT",
            ),
            nfhs=_read_databricks_table(
                "nfhs_5_district_health_indicators",
                "NFHS_TABLE",
                nfhs_columns,
                "NFHS_ROW_LIMIT",
            ),
        ),
        ["Databricks SQL"],
    )


def _load_from_local_csv() -> tuple[Datasets, list[str]]:
    return (
        Datasets(
            facilities=_read_local_csv("facilities.csv"),
            pincodes=_read_local_csv("india_post_pincode_directory.csv"),
            nfhs=_read_local_csv("nfhs_5_district_health_indicators.csv"),
        ),
        ["local CSV fallback"],
    )


@lru_cache(maxsize=1)
def load_datasets() -> tuple[Datasets, list[str]]:
    notes: list[str] = []
    if _databricks_configured():
        try:
            datasets, source_notes = _load_from_databricks()
            return datasets, source_notes
        except Exception as exc:
            notes.append(f"Databricks load failed: {exc}")

    datasets, source_notes = _load_from_local_csv()
    return datasets, notes + source_notes
