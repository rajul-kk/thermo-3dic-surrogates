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
# Pinned: the commit every dataset and tool uses (src/simulators/ice_simulator.py, PINNED_ICE_COMMIT), not upstream HEAD.
# The last command fails the build if the fetched commit is not the one asked for.
ARG ICE_COMMIT=4953952a1ef6d38807ff307212a6f15e5b2ef935
RUN git init /opt/3d-ice \
    && git -C /opt/3d-ice remote add origin https://github.com/esl-epfl/3d-ice \
    && git -C /opt/3d-ice fetch --depth 1 origin ${ICE_COMMIT} \
    && git -C /opt/3d-ice checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/3d-ice rev-parse HEAD)" = "${ICE_COMMIT}"

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
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the compiled emulator binary
COPY --from=builder /opt/3d-ice/bin/3D-ICE-Emulator /opt/3d-ice/bin/3D-ICE-Emulator
RUN chmod +x /opt/3d-ice/bin/3D-ICE-Emulator
# verify_ice_executable() reads the build's commit with `git -C /opt/3d-ice rev-parse HEAD`, so the runtime image needs
# git and the (shallow) repository next to bin/; without them the commit reads 'unknown' and every solve is refused.
COPY --from=builder /opt/3d-ice/.git /opt/3d-ice/.git

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
# THERMO_ICE_EXECUTABLE is what src/simulators/ice_simulator.py (DEFAULT_ICE_EXECUTABLE) reads; app/main.py also honours
# ICE_EXECUTABLE, so both name the same binary.
ENV THERMO_ICE_EXECUTABLE=/opt/3d-ice/bin/3D-ICE-Emulator
ENV ICE_EXECUTABLE=/opt/3d-ice/bin/3D-ICE-Emulator
ENV HOST=0.0.0.0
ENV PORT=8000
ENV RELOAD=false

EXPOSE 8000

CMD ["python3", "run_app.py"]
