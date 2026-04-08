cmake_minimum_required(VERSION 3.22)
project(planner_cvxopt)

set(CMAKE_CXX_STANDARD 20)

# Find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(SpireMS)
ament_auto_find_build_dependencies()

# Add library
add_library(traj_planner SHARED IMPORTED)
target_include_directories(traj_planner INTERFACE ${PROJECT_SOURCE_DIR}/external/traj_planner/include)
set_target_properties(traj_planner PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/external/traj_planner/share/libtraj_planner.so)

# acados OCP solver (pre-built shared library)
add_library(acados_ocp_solver SHARED IMPORTED)
set_target_properties(acados_ocp_solver PROPERTIES
  IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/external/acados_ocp/libacados_ocp_solver_point_mass_ode.so
)

# acados core libraries
# Allow users to override ACADOS_DIR via a cache variable. Default to project-relative
# external/acados so clones can provide the directory there, or override with
# -DACADOS_DIR=/path/to/acados when building.
if(NOT DEFINED ACADOS_DIR)
  set(ACADOS_DIR "${PROJECT_SOURCE_DIR}/external/acados" CACHE PATH "Path to acados installation or repository-provided external/acados")
endif()

# Helper to check for acados artifacts
set(ACADOS_LIB "${ACADOS_DIR}/lib/libacados.so")
set(ACADOS_INCLUDE "${ACADOS_DIR}/include/acados/utils/types.h")

if(EXISTS ${ACADOS_LIB} AND EXISTS ${ACADOS_INCLUDE})
  add_library(acados SHARED IMPORTED)
  set_target_properties(acados PROPERTIES IMPORTED_LOCATION ${ACADOS_LIB})

  if(EXISTS "${ACADOS_DIR}/lib/libhpipm.so")
    add_library(hpipm SHARED IMPORTED)
    set_target_properties(hpipm PROPERTIES IMPORTED_LOCATION ${ACADOS_DIR}/lib/libhpipm.so)
  endif()

  if(EXISTS "${ACADOS_DIR}/lib/libblasfeo.so")
    add_library(blasfeo SHARED IMPORTED)
    set_target_properties(blasfeo PROPERTIES IMPORTED_LOCATION ${ACADOS_DIR}/lib/libblasfeo.so)
  endif()
else()
  message(WARNING "ACADOS not found at ${ACADOS_DIR}. If you need acados support, set -DACADOS_DIR=/path/to/acados or populate ${PROJECT_SOURCE_DIR}/external/acados with headers/libs.")
endif()

# Build
ament_auto_add_executable(${PROJECT_NAME}
  src/main.cpp
  src/planner.cpp
  src/remote_server.cpp
  src/timer.cpp
  src/sampling_planner.cpp
  src/optim_planner.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/external/acados_ocp
)

# Only add ACADOS include dirs and link libraries if the acados IMPORTED target exists
if(TARGET acados)
  if(EXISTS "${ACADOS_DIR}/include")
    target_include_directories(${PROJECT_NAME} PRIVATE ${ACADOS_DIR}/include)
  endif()
  if(EXISTS "${ACADOS_DIR}/include/blasfeo/include")
    target_include_directories(${PROJECT_NAME} PRIVATE ${ACADOS_DIR}/include/blasfeo/include)
  endif()
  if(EXISTS "${ACADOS_DIR}/include/hpipm/include")
    target_include_directories(${PROJECT_NAME} PRIVATE ${ACADOS_DIR}/include/hpipm/include)
  endif()

  target_link_libraries(${PROJECT_NAME} traj_planner Eigen3::Eigen acados_ocp_solver acados hpipm blasfeo)
else()
  # Fall back: link only the libraries that are guaranteed to be present
  target_link_libraries(${PROJECT_NAME} traj_planner Eigen3::Eigen acados_ocp_solver)
  message(WARNING "ACADOS targets not found; building without linking acados/hpipm/blasfeo.
Set -DACADOS_DIR=/path/to/acados or populate ${PROJECT_SOURCE_DIR}/external/acados to enable full functionality.")
endif()


if(SpireMS_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${SpireMS_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${SpireMS_LIBS})
  target_compile_definitions(${PROJECT_NAME} PRIVATE -DUSE_SMS)
endif()

# Install
ament_auto_package(
  INSTALL_TO_SHARE config launch
  USE_SCOPED_HEADER_INSTALL_DIR
)

install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/external/traj_planner/share/
  DESTINATION lib/
)

install(
  FILES ${PROJECT_SOURCE_DIR}/external/acados_ocp/libacados_ocp_solver_point_mass_ode.so
  DESTINATION lib/
)

if(EXISTS "${ACADOS_DIR}/lib")
  install(
    DIRECTORY ${ACADOS_DIR}/lib/
    DESTINATION lib/
    FILES_MATCHING PATTERN "*.so*"
  )
endif()
