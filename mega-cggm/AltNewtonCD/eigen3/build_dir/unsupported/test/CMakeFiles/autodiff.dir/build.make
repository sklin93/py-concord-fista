# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = "/Applications/CMake 2.8-12.app/Contents/bin/cmake"

# The command to remove a file.
RM = "/Applications/CMake 2.8-12.app/Contents/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = "/Applications/CMake 2.8-12.app/Contents/bin/ccmake"

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /usr/local/include/eigen3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /usr/local/include/eigen3/build_dir

# Include any dependencies generated for this target.
include unsupported/test/CMakeFiles/autodiff.dir/depend.make

# Include the progress variables for this target.
include unsupported/test/CMakeFiles/autodiff.dir/progress.make

# Include the compile flags for this target's objects.
include unsupported/test/CMakeFiles/autodiff.dir/flags.make

unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o: unsupported/test/CMakeFiles/autodiff.dir/flags.make
unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o: ../unsupported/test/autodiff.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /usr/local/include/eigen3/build_dir/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o"
	cd /usr/local/include/eigen3/build_dir/unsupported/test && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/autodiff.dir/autodiff.cpp.o -c /usr/local/include/eigen3/unsupported/test/autodiff.cpp

unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/autodiff.dir/autodiff.cpp.i"
	cd /usr/local/include/eigen3/build_dir/unsupported/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /usr/local/include/eigen3/unsupported/test/autodiff.cpp > CMakeFiles/autodiff.dir/autodiff.cpp.i

unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/autodiff.dir/autodiff.cpp.s"
	cd /usr/local/include/eigen3/build_dir/unsupported/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /usr/local/include/eigen3/unsupported/test/autodiff.cpp -o CMakeFiles/autodiff.dir/autodiff.cpp.s

unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.requires:
.PHONY : unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.requires

unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.provides: unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.requires
	$(MAKE) -f unsupported/test/CMakeFiles/autodiff.dir/build.make unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.provides.build
.PHONY : unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.provides

unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.provides.build: unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o

# Object files for target autodiff
autodiff_OBJECTS = \
"CMakeFiles/autodiff.dir/autodiff.cpp.o"

# External object files for target autodiff
autodiff_EXTERNAL_OBJECTS =

unsupported/test/autodiff: unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o
unsupported/test/autodiff: unsupported/test/CMakeFiles/autodiff.dir/build.make
unsupported/test/autodiff: unsupported/test/CMakeFiles/autodiff.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable autodiff"
	cd /usr/local/include/eigen3/build_dir/unsupported/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/autodiff.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
unsupported/test/CMakeFiles/autodiff.dir/build: unsupported/test/autodiff
.PHONY : unsupported/test/CMakeFiles/autodiff.dir/build

unsupported/test/CMakeFiles/autodiff.dir/requires: unsupported/test/CMakeFiles/autodiff.dir/autodiff.cpp.o.requires
.PHONY : unsupported/test/CMakeFiles/autodiff.dir/requires

unsupported/test/CMakeFiles/autodiff.dir/clean:
	cd /usr/local/include/eigen3/build_dir/unsupported/test && $(CMAKE_COMMAND) -P CMakeFiles/autodiff.dir/cmake_clean.cmake
.PHONY : unsupported/test/CMakeFiles/autodiff.dir/clean

unsupported/test/CMakeFiles/autodiff.dir/depend:
	cd /usr/local/include/eigen3/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /usr/local/include/eigen3 /usr/local/include/eigen3/unsupported/test /usr/local/include/eigen3/build_dir /usr/local/include/eigen3/build_dir/unsupported/test /usr/local/include/eigen3/build_dir/unsupported/test/CMakeFiles/autodiff.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : unsupported/test/CMakeFiles/autodiff.dir/depend

