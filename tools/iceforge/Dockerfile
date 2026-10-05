# iceforge + 3D-ICE (built from source). Build:  docker build -t iceforge:latest tools/iceforge
# Use:    docker run --rm -v "$PWD":/work -w /work iceforge:latest check model.stk
#         (the docker backend of `iceforge run` mounts the stk's directory and calls the emulator
#          directly: --entrypoint /opt/3d-ice/bin/3D-ICE-Emulator)
# Build recipe follows the repository's root Dockerfile.

FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran make git bison flex unzip libopenblas-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/esl-epfl/3d-ice /opt/3d-ice
WORKDIR /opt/3d-ice
RUN bash install-superlumt.sh
RUN make

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgomp1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/3d-ice/bin/3D-ICE-Emulator /opt/3d-ice/bin/3D-ICE-Emulator
RUN chmod +x /opt/3d-ice/bin/3D-ICE-Emulator
ENV ICE_EXECUTABLE=/opt/3d-ice/bin/3D-ICE-Emulator
WORKDIR /opt/iceforge
COPY pyproject.toml README.md ./
COPY iceforge ./iceforge
RUN pip3 install --no-cache-dir .
WORKDIR /work
ENTRYPOINT ["iceforge"]
CMD ["--help"]
