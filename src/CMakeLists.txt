cmake_minimum_required(VERSION 3.16)
project(aelvonlock-shc VERSION 2.0.0 LANGUAGES C CXX)

# ─── C Library ─────────────────────────────────────────────────
add_library(aelvonlock_c
    STATIC
    aelvonlock_c/src/aelvonlock.c
)

target_include_directories(aelvonlock_c
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/aelvonlock_c/include
)

if(WIN32)
    target_link_libraries(aelvonlock_c PRIVATE bcrypt)
else()
    find_package(OpenSSL REQUIRED)
    target_link_libraries(aelvonlock_c PRIVATE OpenSSL::Crypto)
    target_compile_definitions(aelvonlock_c PRIVATE HAVE_OPENSSL)
endif()

target_compile_options(aelvonlock_c PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-std=c11 -Wall -Wextra -Wpedantic -Werror>
)

# ─── C Library Test ────────────────────────────────────────────
add_executable(test_aelvonlock_c
    aelvonlock_c/test/test_aelvonlock.c
)

target_link_libraries(test_aelvonlock_c PRIVATE aelvonlock_c)

if(WIN32)
    target_link_libraries(test_aelvonlock_c PRIVATE bcrypt)
else()
    target_link_libraries(test_aelvonlock_c PRIVATE OpenSSL::Crypto)
endif()

# ─── C++ Library (Header-only, but with compiled components) ───
add_library(aelvonlock_cpp INTERFACE)

target_include_directories(aelvonlock_cpp
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/aelvonlock_cpp/include
)

target_compile_features(aelvonlock_cpp INTERFACE cxx_std_20)

if(WIN32)
    target_link_libraries(aelvonlock_cpp INTERFACE bcrypt)
else()
    find_package(OpenSSL REQUIRED)
    target_link_libraries(aelvonlock_cpp INTERFACE OpenSSL::Crypto)
    target_compile_definitions(aelvonlock_cpp INTERFACE HAVE_OPENSSL)
endif()

# ─── C++ Library Test ──────────────────────────────────────────
add_executable(test_aelvonlock_cpp
    aelvonlock_cpp/test/test_aelvonlock.cpp
)

target_link_libraries(test_aelvonlock_cpp PRIVATE aelvonlock_cpp)

if(WIN32)
    target_link_libraries(test_aelvonlock_cpp PRIVATE bcrypt)
else()
    target_link_libraries(test_aelvonlock_cpp PRIVATE OpenSSL::Crypto)
endif()

# ─── Install ───────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS aelvonlock_c
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/aelvonlock_c/include/aelvonlock.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/aelvonlock_cpp/include/aelvonlock.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
