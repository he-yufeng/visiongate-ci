ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY cv5-requirements.txt /tmp/cv5-requirements.txt
RUN python -m pip install --no-deps --require-hashes -r /tmp/cv5-requirements.txt --target ${LAMBDA_TASK_ROOT}
COPY visiongate_ci ${LAMBDA_TASK_ROOT}/visiongate_ci
COPY contracts ${LAMBDA_TASK_ROOT}/contracts
COPY cloud ${LAMBDA_TASK_ROOT}/cloud
CMD ["cloud.handler.lambda_handler"]
