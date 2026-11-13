# Third-party notices

wmnav contains code derived from the project below. Its license requires the
copyright and permission notice to be kept with the code; it is reproduced here.

| project | used in | license |
|---|---|---|
| [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) (Autonomous Robots Lab, NTNU) | `wmnav/env/configs/` (sim, robot, assets), `wmnav/env/patches.py` | BSD 3-Clause |

sample-factory, Isaac Gym and aerial_gym are used as installed dependencies and are
not redistributed here. The DreamerV3 implementation in `wmnav/dreamer/` is
written for this project; the method is that of Hafner et al. (see the README).

## Aerial Gym Simulator

```
BSD 3-Clause License

Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
