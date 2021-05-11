# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen

# Utility rule file for tetgen-download.

# Include the progress variables for this target.
include CMakeFiles/tetgen-download.dir/progress.make

CMakeFiles/tetgen-download: CMakeFiles/tetgen-download-complete


CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-install
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-mkdir
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-download
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-update
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-patch
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-configure
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-build
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-install
CMakeFiles/tetgen-download-complete: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'tetgen-download'"
	/usr/bin/cmake -E make_directory /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles
	/usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles/tetgen-download-complete
	/usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-done

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-install: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'tetgen-download'"
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E echo_append
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-install

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'tetgen-download'"
	/usr/bin/cmake -E make_directory /mnt/school/shapeMemory/submodules/libigl/cmake/../external/tetgen
	/usr/bin/cmake -E make_directory /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build
	/usr/bin/cmake -E make_directory /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix
	/usr/bin/cmake -E make_directory /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/tmp
	/usr/bin/cmake -E make_directory /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp
	/usr/bin/cmake -E make_directory /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src
	/usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-mkdir

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-gitinfo.txt
tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'tetgen-download'"
	cd /mnt/school/shapeMemory/submodules/libigl/external && /usr/bin/cmake -P /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/tmp/tetgen-download-gitclone.cmake
	cd /mnt/school/shapeMemory/submodules/libigl/external && /usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-download

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-update: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing update step for 'tetgen-download'"
	cd /mnt/school/shapeMemory/submodules/libigl/external/tetgen && /usr/bin/cmake -P /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/tmp/tetgen-download-gitupdate.cmake

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-patch: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'tetgen-download'"
	/usr/bin/cmake -E echo_append
	/usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-patch

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-configure: tetgen-download-prefix/tmp/tetgen-download-cfgcmd.txt
tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-configure: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-update
tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-configure: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'tetgen-download'"
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E echo_append
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-configure

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-build: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'tetgen-download'"
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E echo_append
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-build

tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-test: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'tetgen-download'"
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E echo_append
	cd /mnt/school/shapeMemory/submodules/libigl/build/tetgen-build && /usr/bin/cmake -E touch /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-test

tetgen-download: CMakeFiles/tetgen-download
tetgen-download: CMakeFiles/tetgen-download-complete
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-install
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-mkdir
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-download
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-update
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-patch
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-configure
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-build
tetgen-download: tetgen-download-prefix/src/tetgen-download-stamp/tetgen-download-test
tetgen-download: CMakeFiles/tetgen-download.dir/build.make

.PHONY : tetgen-download

# Rule to build all files generated by this target.
CMakeFiles/tetgen-download.dir/build: tetgen-download

.PHONY : CMakeFiles/tetgen-download.dir/build

CMakeFiles/tetgen-download.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tetgen-download.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tetgen-download.dir/clean

CMakeFiles/tetgen-download.dir/depend:
	cd /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen /mnt/school/shapeMemory/submodules/libigl/external/.cache/tetgen/CMakeFiles/tetgen-download.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tetgen-download.dir/depend

