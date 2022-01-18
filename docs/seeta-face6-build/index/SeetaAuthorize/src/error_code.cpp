#include "error_code.h"

#include <map>


static std::map<int, std::string> error_msgs =
{
    {10000,   "Model authorization failed"},
    {10001,   "Read model file failed"},
    {10002,   "Open file error,"},
    {10003,   "Get an illegal file,"},
};


std::string error_str( int error_code )
{
    std::string str, strret;
    std::map<int, std::string>::iterator iter = error_msgs.find( error_code );
    if( iter != error_msgs.end() )
    {
        str = iter->second;
    }
    else
    {
        return "";
    }

    strret = "error code:";
    strret += std::to_string( error_code ) + "\n";
    strret += "\tdescription:";
    strret += str;
    return strret;
}
