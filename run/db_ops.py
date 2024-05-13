import os
from snowflake import connector


def snowflake_connection():
    return connector.connect(
        user=os.environ['PDM_SF_USER'],
        password=os.environ['PDM_SF_PASSWORD'],
        account='fbb.us-east-1',
        warehouse='DS_ANALYTICS',
        database=os.environ.get('PDM_SF_DB', 'DS_PROJECTS'),
        schema=os.environ.get('PDM_SF_SCHEMA', 'PRODUCT_DATA_MODEL'),
        role=os.environ.get('PDM_SF_ROLE', 'PRODUCT_DATA_MODEL')
    )


def query_db(query):
    with snowflake_connection() as conn:
        with conn.cursor() as cur:
            return cur.execute(query).fetch_pandas_all()


