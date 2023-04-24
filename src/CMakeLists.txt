# Get a list of all source files in the directory
file(GLOB SOURCES *.cpp)

include_directories(${CMAKE_SOURCE_DIR}/../includes)

# Add the source files
add_library(mylibrary
  ${SOURCES}
  )


# Link against any required libraries
target_link_libraries(mylibrary PRIVATE
  ${TORCH_LIBRARIES}
  )

# Set include directories for the target
target_include_directories(mylibrary PUBLIC
  ../include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_INCLUDE_DIRS}
)
