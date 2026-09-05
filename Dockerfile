FROM public.ecr.aws/lambda/python:3.14-experimental.2026.08.29.12-x86_64

COPY src/ ${LAMBDA_TASK_ROOT}/src

COPY api_modules.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/api_modules.txt

CMD [ "src.api.endpoint.handler" ]