# ── Stage 1: compile SuperLU MT + 3D-ICE ─────────────────────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran make git \
    bison flex unzip \
    libopenblas-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone 3D-ICE (includes superlu_mt-4.0.0.zip in-repo)
RUN git clone --depth 1 https://github.com/esl-epfl/3d-ice /opt/3d-ice

WORKDIR /opt/3d-ice

# Build SuperLU MT using the bundled script
RUN bash install-superlumt.sh

# Build 3D-ICE (all → lib + bin; emulator ends up in bin/)
RUN make


# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgomp1 libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the compiled emulator binary
COPY --from=builder /opt/3d-ice/bin/3D-ICE-Emulator /opt/3d-ice/bin/3D-ICE-Emulator
RUN chmod +x /opt/3d-ice/bin/3D-ICE-Emulator

WORKDIR /app

# Install Python dependencies in two layers so routine app changes
# don't bust the heavier pipeline layer cache
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    numpy scipy pyyaml matplotlib seaborn pytest

COPY requirements_app.txt .
RUN pip3 install --no-cache-dir -r requirements_app.txt

# Copy source
COPY src/  ./src/
COPY app/  ./app/
COPY run_app.py .

# Runtime config
ENV ICE_EXECUTABLE=/opt/3d-ice/bin/3D-ICE-Emulator
ENV HOST=0.0.0.0
ENV PORT=8000
ENV RELOAD=false

EXPOSE 8000

CMD ["python3", "run_app.py"]
