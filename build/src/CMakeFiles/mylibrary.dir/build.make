# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chandler/src/deep-q-network

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chandler/src/deep-q-network/build

# Include any dependencies generated for this target.
include src/CMakeFiles/mylibrary.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/mylibrary.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/mylibrary.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/mylibrary.dir/flags.make

src/CMakeFiles/mylibrary.dir/DQN.cpp.o: src/CMakeFiles/mylibrary.dir/flags.make
src/CMakeFiles/mylibrary.dir/DQN.cpp.o: ../src/DQN.cpp
src/CMakeFiles/mylibrary.dir/DQN.cpp.o: src/CMakeFiles/mylibrary.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chandler/src/deep-q-network/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/mylibrary.dir/DQN.cpp.o"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/mylibrary.dir/DQN.cpp.o -MF CMakeFiles/mylibrary.dir/DQN.cpp.o.d -o CMakeFiles/mylibrary.dir/DQN.cpp.o -c /home/chandler/src/deep-q-network/src/DQN.cpp

src/CMakeFiles/mylibrary.dir/DQN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mylibrary.dir/DQN.cpp.i"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chandler/src/deep-q-network/src/DQN.cpp > CMakeFiles/mylibrary.dir/DQN.cpp.i

src/CMakeFiles/mylibrary.dir/DQN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mylibrary.dir/DQN.cpp.s"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chandler/src/deep-q-network/src/DQN.cpp -o CMakeFiles/mylibrary.dir/DQN.cpp.s

src/CMakeFiles/mylibrary.dir/Environment.cpp.o: src/CMakeFiles/mylibrary.dir/flags.make
src/CMakeFiles/mylibrary.dir/Environment.cpp.o: ../src/Environment.cpp
src/CMakeFiles/mylibrary.dir/Environment.cpp.o: src/CMakeFiles/mylibrary.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chandler/src/deep-q-network/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/mylibrary.dir/Environment.cpp.o"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/mylibrary.dir/Environment.cpp.o -MF CMakeFiles/mylibrary.dir/Environment.cpp.o.d -o CMakeFiles/mylibrary.dir/Environment.cpp.o -c /home/chandler/src/deep-q-network/src/Environment.cpp

src/CMakeFiles/mylibrary.dir/Environment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mylibrary.dir/Environment.cpp.i"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chandler/src/deep-q-network/src/Environment.cpp > CMakeFiles/mylibrary.dir/Environment.cpp.i

src/CMakeFiles/mylibrary.dir/Environment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mylibrary.dir/Environment.cpp.s"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chandler/src/deep-q-network/src/Environment.cpp -o CMakeFiles/mylibrary.dir/Environment.cpp.s

src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o: src/CMakeFiles/mylibrary.dir/flags.make
src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o: ../src/ReplayMemory.cpp
src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o: src/CMakeFiles/mylibrary.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chandler/src/deep-q-network/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o -MF CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o.d -o CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o -c /home/chandler/src/deep-q-network/src/ReplayMemory.cpp

src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mylibrary.dir/ReplayMemory.cpp.i"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chandler/src/deep-q-network/src/ReplayMemory.cpp > CMakeFiles/mylibrary.dir/ReplayMemory.cpp.i

src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mylibrary.dir/ReplayMemory.cpp.s"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chandler/src/deep-q-network/src/ReplayMemory.cpp -o CMakeFiles/mylibrary.dir/ReplayMemory.cpp.s

src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o: src/CMakeFiles/mylibrary.dir/flags.make
src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o: ../src/TicTacToeEnvironment.cpp
src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o: src/CMakeFiles/mylibrary.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chandler/src/deep-q-network/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o -MF CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o.d -o CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o -c /home/chandler/src/deep-q-network/src/TicTacToeEnvironment.cpp

src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.i"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chandler/src/deep-q-network/src/TicTacToeEnvironment.cpp > CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.i

src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.s"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chandler/src/deep-q-network/src/TicTacToeEnvironment.cpp -o CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.s

src/CMakeFiles/mylibrary.dir/main.cpp.o: src/CMakeFiles/mylibrary.dir/flags.make
src/CMakeFiles/mylibrary.dir/main.cpp.o: ../src/main.cpp
src/CMakeFiles/mylibrary.dir/main.cpp.o: src/CMakeFiles/mylibrary.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chandler/src/deep-q-network/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/mylibrary.dir/main.cpp.o"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/mylibrary.dir/main.cpp.o -MF CMakeFiles/mylibrary.dir/main.cpp.o.d -o CMakeFiles/mylibrary.dir/main.cpp.o -c /home/chandler/src/deep-q-network/src/main.cpp

src/CMakeFiles/mylibrary.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mylibrary.dir/main.cpp.i"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chandler/src/deep-q-network/src/main.cpp > CMakeFiles/mylibrary.dir/main.cpp.i

src/CMakeFiles/mylibrary.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mylibrary.dir/main.cpp.s"
	cd /home/chandler/src/deep-q-network/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chandler/src/deep-q-network/src/main.cpp -o CMakeFiles/mylibrary.dir/main.cpp.s

# Object files for target mylibrary
mylibrary_OBJECTS = \
"CMakeFiles/mylibrary.dir/DQN.cpp.o" \
"CMakeFiles/mylibrary.dir/Environment.cpp.o" \
"CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o" \
"CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o" \
"CMakeFiles/mylibrary.dir/main.cpp.o"

# External object files for target mylibrary
mylibrary_EXTERNAL_OBJECTS =

src/libmylibrary.a: src/CMakeFiles/mylibrary.dir/DQN.cpp.o
src/libmylibrary.a: src/CMakeFiles/mylibrary.dir/Environment.cpp.o
src/libmylibrary.a: src/CMakeFiles/mylibrary.dir/ReplayMemory.cpp.o
src/libmylibrary.a: src/CMakeFiles/mylibrary.dir/TicTacToeEnvironment.cpp.o
src/libmylibrary.a: src/CMakeFiles/mylibrary.dir/main.cpp.o
src/libmylibrary.a: src/CMakeFiles/mylibrary.dir/build.make
src/libmylibrary.a: src/CMakeFiles/mylibrary.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chandler/src/deep-q-network/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libmylibrary.a"
	cd /home/chandler/src/deep-q-network/build/src && $(CMAKE_COMMAND) -P CMakeFiles/mylibrary.dir/cmake_clean_target.cmake
	cd /home/chandler/src/deep-q-network/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mylibrary.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/mylibrary.dir/build: src/libmylibrary.a
.PHONY : src/CMakeFiles/mylibrary.dir/build

src/CMakeFiles/mylibrary.dir/clean:
	cd /home/chandler/src/deep-q-network/build/src && $(CMAKE_COMMAND) -P CMakeFiles/mylibrary.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/mylibrary.dir/clean

src/CMakeFiles/mylibrary.dir/depend:
	cd /home/chandler/src/deep-q-network/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chandler/src/deep-q-network /home/chandler/src/deep-q-network/src /home/chandler/src/deep-q-network/build /home/chandler/src/deep-q-network/build/src /home/chandler/src/deep-q-network/build/src/CMakeFiles/mylibrary.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/mylibrary.dir/depend
