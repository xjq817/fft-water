# FFT-ocean-water

MAC OS:
`brew info package_name` to check its path, and update the path in CMakeList.txt

To run the code:

```
mkdir build/
cd build/
cmake ..
make
./main
```

To make a video:

```
ffmpeg -r 30 -start_number 0 -i ../result/output/%04d.bmp -vcodec mpeg4 -b:v 30M -s 800x600 ../result/video.mp4
```

- Change the skybox:

  Copy the maps in `/image/Map/`  named `nx,ny,nz,px,py,pz` into 	`image/` directory.

- Change the wind speed:

  change the `windSpeed` in `main.cpp`

- Change the spray effect.

  change $\epsilon= 0.025$ of  `Jaccobi = glm::max(Jaccobi - 0.025f, 0.f);` in function `writeFoldingMap` in `ocean.cpp`.

  There will be more whitecaps with larger $\epsilon$ and fewer whitecaps with smaller $\epsilon$.

- Change fresnel effect

  In `shader/fsOcean.glsl`, there are two types of fresnel implement

  - with texture: `float fresnel = texture(texFresnel, fresUv).r * 0.5;`
  - with Schlick approximation: `float fresnel = fresnelSchlick(max(dot(H, V), 0.0), 0.02);`

