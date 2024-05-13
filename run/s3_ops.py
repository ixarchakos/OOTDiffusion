import boto3
import mimetypes
import os


def s3_client():
    s3 = boto3.client('s3')
    return s3


def upload_to_s3(filename, laydown_image, s3_client, s3_bucket, s3_path):
    content_type, _ = mimetypes.guess_type(filename)

    s3_client.upload_fileobj(
        laydown_image,
        s3_bucket,
        os.path.join(s3_path, filename),
        ExtraArgs={
            "ContentType": content_type
        }
    )


def upload_file(s3, laydown_image, s3_path, filename):
    upload_to_s3(
        filename=filename,
        laydown_image=laydown_image,
        s3_bucket=os.environ['AWS_S3_BUCKET'],
        s3_client=s3,
        s3_path=s3_path
    )

    return f"https://{os.environ['AWS_S3_BUCKET']}.s3.amazonaws.com/{s3_path}/{filename}"
