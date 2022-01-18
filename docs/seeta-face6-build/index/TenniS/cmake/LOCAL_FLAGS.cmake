# set flags

if (MSVC)
    include(FLAGS_MSVC)
else()
    include(FLAGS_GCC)
endif()
