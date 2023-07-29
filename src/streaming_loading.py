import logging
import asyncio
from azure.eventhub.aio import EventHubConsumerClient
import os
from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/camivasz/almond-datathon-ffcfe3899e67.json"


CONNECTION_STRING = "Endpoint=sb://factored-datathon.servicebus.windows.net/;SharedAccessKeyName=datathon_group_3;SharedAccessKey=JLEggz9GNlDdLvbypDAudzTABp+WnVeIY+AEhBAupi4=;EntityPath=factored_datathon_amazon_reviews_3"
EVENT_HUB_LISTEN_POLICY_KEY = "sJJnyi8GGTBAa55jY89kacoT6hXAzWx2B+AEhCPEKYE="
CONSUMER_GROUP = 'almond'
EVENT_HUB_NAME = "factored_datathon_amazon_reviews_3"

logger = logging.getLogger("azure.eventhub")
logging.basicConfig(level=logging.INFO)

async def on_event(partition_context, event):
    filename = f"{partition_context.partition_id}_{event.sequence_number}.json"
    source_file_name = f"reads/{filename}"
    destination_blob_name = f"patition_0/{filename}"
    with open(f"reads/{filename}", 'wb') as fp:
        fp.write(next(event.body))
    if event.sequence_number > 15391:
        client_storage = storage.Client()
        bucket_name = "amazon-reviews-almond-3"
        bucket = client_storage.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
    logger.info("Received event {} from partition {}".format(event.sequence_number, partition_context.partition_id))
    await partition_context.update_checkpoint(event)

async def receive():
    client = EventHubConsumerClient.from_connection_string(CONNECTION_STRING, 
                                                            CONSUMER_GROUP, 
                                                            eventhub_name=EVENT_HUB_NAME)
    async with client:
        await client.receive(
            on_event=on_event,
            starting_position="-1",  # "-1" is from the beginning of the partition.
        )

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(receive())