# base cmake tools

# ts_add_library(<name> [STATIC | SHARED | MODULE] source1 [source2 ...])
function(ts_add_library target_name target_type)
    # get all src_files
    set(src_files)
    set(INDEX 2)
    while(INDEX LESS ${ARGC})
        list(APPEND src_files ${ARGV${INDEX}})
        math(EXPR INDEX "${INDEX} + 1")
    endwhile()

    # add library target
    if (TS_USE_CUDA)
        cuda_add_library(${target_name} ${target_type} ${src_files})
    else()
        add_library(${target_name} ${target_type} ${src_files})
    endif()
endfunction()

# ts_add_executable(<name> source1 [source2 ...])
function(ts_add_executable target_name)
    # get all src_files
    set(src_files)
    set(INDEX 1)
    while(INDEX LESS ${ARGC})
        list(APPEND src_files ${ARGV${INDEX}})
        math(EXPR INDEX "${INDEX} + 1")
    endwhile()

    # add library target
    if (TS_USE_CUDA)
        cuda_add_executable(${target_name} ${src_files})
    else()
        add_executable(${target_name} ${src_files})
    endif()
endfunction()

# ts_add_instruction_support(<target> flag)
# flag:
# 0:add avx,fma support
# 1:add avx support
# 2:add sse support
function(ts_add_instruction_support target_name flag)
    if (MSVC)
        if(${flag} EQUAL 0)
            message(STATUS "[Info] target:${target_name} support avx and fma")
            target_compile_options(${target_name} PRIVATE /arch:AVX)
        elseif(${flag} EQUAL 1)
            message(STATUS "[Info] target:${target_name} support avx")
            target_compile_options(${target_name} PRIVATE /arch:AVX)
        elseif(${flag} EQUAL 2)
            message(STATUS "[Info] target:${target_name} support sse")
        endif()
    else()
        if(${flag} EQUAL 0)
            message(STATUS "[Info] target:${target_name} support avx and fma")
            target_compile_options(${target_name} PRIVATE -mavx -mavx2 -mfma)
        elseif(${flag} EQUAL 1)
            message(STATUS "[Info] target:${target_name} support avx")
            target_compile_options(${target_name} PRIVATE -mavx -mavx2)
        elseif(${flag} EQUAL 2)
            message(STATUS "[Info] target:${target_name} support sse")
            target_compile_options(${target_name} PRIVATE -msse2)
        endif()
    endif()
endfunction()