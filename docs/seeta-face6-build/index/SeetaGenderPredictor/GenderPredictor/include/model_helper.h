#ifndef MODELHELPER_H
#define MODELHELPER_H

#include <string>
#include <orz/utils/log.h>
#include <orz/io/jug/jug.h>

#ifdef SEETA_MODEL_ENCRYPT
#include <hidden/SeetaLockFunction.h>
#else
#include <orz/io/stream/modelfilestream.h>
#endif



#ifdef SEETA_MODEL_ENCRYPT
/**
 * @param modelfile the model file path, support json, sta, csta type model file.
 * @return return the model's jub object
*/
orz::jug get_model_jug(const char * modelfile) {
    return SeetaLock_GetModelJug(modelfile);
}

/**
 * @param in the input stream, support json, sta, csta type model input stream.
 * @return return the model's jub object
*/
orz::jug get_model_jug(orz::InputStream * in) {
    return SeetaLock_GetModelJug(in);
}

 
#else
/**
 * @param modelfile the model file path, support json, sta type model file.
 * @return return the model's jub object
*/
orz::jug get_model_jug(const char * modelfile) {
    std::string strfile(modelfile);
    orz::ModelFileInputStream stream(strfile);
    return stream.read_jug();
}

/**
 * @param in the input stream, support json, sta type model input stream.
 * @return return the model's jub object
*/
orz::jug get_model_jug(orz::InputStream * in) {
    auto stream = std::shared_ptr<orz::InputStream>(in, [](orz::InputStream *p) { });
    return orz::ModelFileInputStream::read_jug(stream);
}


#endif

#endif
