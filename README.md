# FFT-ocean-water
MACOS:
`brew info package_name` to check its path

To run the code:

```
mkdir build/
cd build/
cmake ..
make
./main
```
To make vide:
```
mkdir result/output/
make video
```

or
```
ffmpeg -r 30 -start_number 0 -i ../result/output/%04d.bmp -vcodec mpeg4 -b:v 30M -s 800x600 ../result/video.mp4
```
