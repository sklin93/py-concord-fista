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
include doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/depend.make

# Include the progress variables for this target.
include doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/progress.make

# Include the compile flags for this target's objects.
include doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/flags.make

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o: doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/flags.make
doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o: doc/snippets/compile_tut_arithmetic_redux_minmax.cpp
doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o: ../doc/snippets/tut_arithmetic_redux_minmax.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /usr/local/include/eigen3/build_dir/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o"
	cd /usr/local/include/eigen3/build_dir/doc/snippets && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o -c /usr/local/include/eigen3/build_dir/doc/snippets/compile_tut_arithmetic_redux_minmax.cpp

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.i"
	cd /usr/local/include/eigen3/build_dir/doc/snippets && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /usr/local/include/eigen3/build_dir/doc/snippets/compile_tut_arithmetic_redux_minmax.cpp > CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.i

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.s"
	cd /usr/local/include/eigen3/build_dir/doc/snippets && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /usr/local/include/eigen3/build_dir/doc/snippets/compile_tut_arithmetic_redux_minmax.cpp -o CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.s

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.requires:
.PHONY : doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.requires

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.provides: doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.requires
	$(MAKE) -f doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/build.make doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.provides.build
.PHONY : doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.provides

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.provides.build: doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o

# Object files for target compile_tut_arithmetic_redux_minmax
compile_tut_arithmetic_redux_minmax_OBJECTS = \
"CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o"

# External object files for target compile_tut_arithmetic_redux_minmax
compile_tut_arithmetic_redux_minmax_EXTERNAL_OBJECTS =

doc/snippets/compile_tut_arithmetic_redux_minmax: doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o
doc/snippets/compile_tut_arithmetic_redux_minmax: doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/build.make
doc/snippets/compile_tut_arithmetic_redux_minmax: doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable compile_tut_arithmetic_redux_minmax"
	cd /usr/local/include/eigen3/build_dir/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/link.txt --verbose=$(VERBOSE)
	cd /usr/local/include/eigen3/build_dir/doc/snippets && ./compile_tut_arithmetic_redux_minmax >/usr/local/include/eigen3/build_dir/doc/snippets/tut_arithmetic_redux_minmax.out

# Rule to build all files generated by this target.
doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/build: doc/snippets/compile_tut_arithmetic_redux_minmax
.PHONY : doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/build

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/requires: doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/compile_tut_arithmetic_redux_minmax.cpp.o.requires
.PHONY : doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/requires

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/clean:
	cd /usr/local/include/eigen3/build_dir/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/cmake_clean.cmake
.PHONY : doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/clean

doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/depend:
	cd /usr/local/include/eigen3/build_dir && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /usr/local/include/eigen3 /usr/local/include/eigen3/doc/snippets /usr/local/include/eigen3/build_dir /usr/local/include/eigen3/build_dir/doc/snippets /usr/local/include/eigen3/build_dir/doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : doc/snippets/CMakeFiles/compile_tut_arithmetic_redux_minmax.dir/depend

