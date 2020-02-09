Standard APIs for AI operations

# Environment Requirement

| Dependency | Version required    |
| ---------- | ------------------- |
| gcc        | `5.0` or higher     |
| CMake      | `3.11` or higher    |

# Setup
You can setup *Standard APIs for AI operations* by following instructions:
1. Use **git clone** instruction to download source code

      ```bash
      git clone https://github.com/pku-hpc/aitisa_api.git
      ```

2. Make a new directory **build** under the project directory, then use **cmake** instruction

      ```bash
      mkdir build  
      cd build  
      cmake ..
      ```

3. Use **make** instruction to compile the code

      ```bash
      make
      ```
      
4. Run testing file. Take running convolution operator testing file as an example

      ```bash
      cd bin
      ./conv_test
      ```
