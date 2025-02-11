docker build -t deephase-image .

docker run -v $(pwd):/app -d -it --name deephase-container deephase-image /bin/bash && docker attach deephase-container