from feast import Entity, FeatureView, Field
from feast.types import Int64, Float32
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

# Define entity
house = Entity(
    name="house_id",
    join_keys=["house_id"],
)

# Define source (points to your table in Postgres)
house_source = PostgreSQLSource(
    name="house_source",
    query="SELECT * FROM house_features",   # or table="house_features"
    timestamp_field="event_timestamp",
)

# Define feature view
house_features = FeatureView(
    name="house_features",
    entities=[house],
    ttl=None,
    schema=[
        Field(name="price", dtype=Float32),
        Field(name="area", dtype=Float32),
        Field(name="bedrooms", dtype=Float32),
        Field(name="mainroad", dtype=Int64),
    ],
    online=True,
    source=house_source,
)
