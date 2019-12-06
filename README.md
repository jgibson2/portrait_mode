You will need various things to compile and use this:

* OpenCV4 + opencv4-contrib

* glog

* Eigen

* nng

* Ceres

If you choose to use the monodepth server, be warned that it has a DIFFERNT LICENSE than this project. You will alos need to install `pynng` and move the script `resources/depth_server.py` into the monodepth2 folder (and, of course, run the server).

Note that this project borrows heavily from `https://github.com/tvandenzegel/fast_bilateral_space_stereo`, with some modifications for speed and stability.

Usage:

`cmake .`

`make -j4`

`bin/portrait_mode <IMAGE_1> <IMAGE_2> <BLUR_STRENGTH> <DEAD_ZONE>`

By default, this program will use the fast bilateral depth solver and a parallelized disc blur.
