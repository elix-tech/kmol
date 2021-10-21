# Docker execution

## Building docker
```
cd docker
docker build -t kmol ..
```

If the build is successful, the torch version will be displayed with the following command (1.6.0)
```
docker run -it kmol
```

For example, if you want to run your script `<fullpath>/run.sh`

```
docker run -v <fullpath>:/mnt -it --entrypoint "/mnt/run.sh"  kmol
```

