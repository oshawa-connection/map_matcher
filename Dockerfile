
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

WORKDIR /hello
COPY MapServer/ /hello/MapServer
COPY mapfiles /mapfiles
RUN mkdir /output


RUN apt update
RUN apt upgrade -y

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common g++ make \
cmake wget curl apache2-dev \
build-essential   openssl autoconf gtk-doc-tools libc-ares-dev libc-ares-dev libcurl4 python3-pip


RUN python3 -m pip install attrs --break-system-packages

# Install mapcache dependencies provided by Ubuntu repositories
RUN apt-get install -y --fix-missing --no-install-recommends \
    libxml2-dev \
    libxslt1-dev \
    libfribidi-dev \
    libcurl4-gnutls-dev \
    libexempi-dev \
    libfcgi-dev \
    libpsl-dev \
    libharfbuzz-dev \
    libexempi-dev \
    libfcgi-dev \
    libproj-dev

WORKDIR /hello/MapServer/build

RUN cmake .. -DWITH_THREAD_SAFETY=1 \
        -DWITH_SOS=1 \
        -DWITH_WMS=1 \
        -DWITH_GIF=0 \
        -DWITH_FRIBIDI=1 \
        -DWITH_HARFBUZZ=1 \
        -DWITH_ICONV=1 \
        -DWITH_GEOS=0 \
        -DWITH_CURL=1 \
        -DWITH_CLIENT_WMS=1 \
        -DWITH_CLIENT_WFS=1 \
        -DWITH_WFS=1 \
        -DWITH_WCS=1 \
        -DWITH_CAIRO=0 \
        -DWITH_LIBXML2=1 \
        -DWITH_POSTGIS=0 \
        -DWITH_EXEMPI=1 \
        -DWITH_XMLMAPFILE=1 \
        -DWITH_PIXMAN=0 \
        -DWITH_PROTOBUFC=0
RUN cmake --build . --target map2img -j 4
RUN ldconfig

ENV PATH="$PATH:/opt/gtk/bin"

