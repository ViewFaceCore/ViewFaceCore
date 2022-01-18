//
// Created by kier on 2018/10/16.
//

#include <module/graph.h>
#include <module/module.h>
#include <global/setup.h>
#include <runtime/workbench.h>
#include <global/operator_factory.h>
#include <utils/ctxmgr.h>
#include <core/tensor_builder.h>
#include <module/menu.h>
#include <utils/box.h>
#include <module/io/fstream.h>
#include <core/tensor_builder.h>

#include <backend/name.h>
#include <cstring>

#include <kernels/cpu/resize2d.h>
#include <kernels/cpu/reshape.h>
#include <kernels/cpu/pooling2d.h>

#include <unordered_map>
#include <fstream>
#include <set>
#include <map>
#include <sstream>

using namespace ts;

std::stringstream g_node_stream;
std::stringstream g_direct_stream;
std::set<std::string> g_retentions;
std::set<std::string> g_set;

struct node_info {
    std::string shape;
    std::string color;
};

std::unordered_map<std::string, node_info> g_node_infos;


////////////////////////////////////////////////////////////
static void init_node_info() {
    g_node_infos[name::layer::field()] = {"box","#05814b"};
    g_node_infos[name::layer::pack()] = {"box","#a6ea27"};
    g_node_infos[name::layer::dimshuffle()] = {"box","#ea7527"};
    g_node_infos[name::layer::transpose()] = {"box","#eac327"};
    g_node_infos[name::layer::reshape()] = {"box","#27eab3"};
    g_node_infos[name::layer::shape()] = {"box","#ea27ac"};

    g_node_infos[name::layer::conv2d()] = {"rect","#e9cd4f"};
    g_node_infos[name::layer::conv2d_v2()] = {"rect","#4fe98d"};
    g_node_infos[name::layer::depthwise_conv2d()] = {"rect","#cf4fe9"};
    g_node_infos[name::layer::depthwise_conv2d_v2()] = {"rect","#e9694f"};
    g_node_infos[name::layer::pad()] = {"mcircle","#967c77"};
    g_node_infos[name::layer::add_bias()] = {"egg","#e7c9c3"};
    g_node_infos[name::layer::batch_norm()] = {"box","#cecb6c"};
    g_node_infos[name::layer::batch_scale()] = {"box","#4d4b1d"};
    g_node_infos[name::layer::fused_batch_norm()] = {"box","#38b062"};
    g_node_infos[name::layer::add()] = {"hexagon","#f5bcbc"};
    g_node_infos[name::layer::sub()] = {"hexagon","#a66767"};
    g_node_infos[name::layer::div()] = {"hexagon","#cf4545"};
    g_node_infos[name::layer::mul()] = {"hexagon","#b9a0a0"};
    g_node_infos[name::layer::inner_prod()] = {"invhouse","#6277ee"};
    g_node_infos[name::layer::relu()] = {"diamond","#82ee62"};
    g_node_infos[name::layer::prelu()] = {"diamond","#449c29"};
    g_node_infos[name::layer::relu_max()] = {"diamond","#578c47"};
    g_node_infos[name::layer::sigmoid()] = {"diamond","#96e07f"};
    g_node_infos[name::layer::softmax()] = {"diamond","#9ccb8e"};
    g_node_infos[name::layer::concat()] = {"invhouse","#53658b"};
    g_node_infos[name::layer::flatten()] = {"rect","#6d6d3b"};
    g_node_infos[name::layer::to_float()] = {"rect","#adddcc"};
    g_node_infos[name::layer::pooling2d()] = {"rect","#cbaddd"};
    g_node_infos[name::layer::pooling2d_v2()] = {"rect","#b663ea"};
    g_node_infos[name::layer::resize2d()] = {"invtrapezium","#2bce93"};
    g_node_infos[name::layer::mx_pooling2d_padding()] = {"rect","#8f9996"};
    g_node_infos[name::layer::copy()] = {"triangle","#904546"};
    g_node_infos["TensorStack"] = {"rect","#99cccc"};
}

static std::string get_node_id(const void * ptr) {
    char buf[100] = {0};

    //offset is 2 not 4, for mark ptr header 0X
#ifdef WIN32 
    _snprintf_s(buf+2, sizeof(buf) - 2, sizeof(buf) - 3, "%p", ptr);
#else
    snprintf(buf+2, sizeof(buf) - 3, "%p", ptr);
#endif

    buf[0] = 'n';
    buf[1] = 'o';
    buf[2] = 'd';
    buf[3] = 'e';
    return std::string(buf);
}

static std::string get_bubble_name(const std::string & name, const void * ptr)
{
    char buf[100] = {0};
#ifdef WIN32 
    _snprintf_s(buf, sizeof(buf), sizeof(buf) - 1, "%p", ptr);
#else
    snprintf(buf, sizeof(buf) - 1, "%p", ptr);
#endif 
 
    std::string str(buf);
    str = name + "_" + str; 
    return str;
}

static bool get_node_info(const std::string &name, node_info & info) {
    std::unordered_map<std::string, node_info>::iterator iter = g_node_infos.find(name);
    if(iter != g_node_infos.end()) {
        info = iter->second;
        return true;
    }
    return false;
}

static std::string get_resize2d_type_str(int type) {
    if(type == 0) {
        return "linear";
    }else {
        return "cublic";
    }
}
static std::string get_pooling_type_str(int type) {
    if(type == 0) {
        return "MAX";
    }else {
        return "AVG";
    }
}


static std::string get_pooling_padding_type_str(int type) {
    if(type == 0) {
        return "BLACK";
    }else if(type == 1) {
        return "COPY";
    }else{
        return "LOOP";
    }
}

static std::string convert_html(const std::string & strsrc) {
    std::string str;
    for(size_t i=0; i<strsrc.length(); i++) {
        if(strsrc[i] == ' ') {
            str += "&nbsp;";
        }else if(strsrc[i] == '<') {
            str += "&lt;";
        }else if(strsrc[i] == '>') {
            str += "&gt;";
        }else if(strsrc[i] == '&') {
            str += "&amp;";
        }else if(strsrc[i] == '\"') {
            str += "&quot;";
        }else {
            str += strsrc[i];
        }
    }
    return str;
}

/*
static void ReplaceAll(std::string& strSource, const std::string& strOld, const std::string& strNew)
{
    size_t nPos = 0;
    while ((nPos = strSource.find(strOld, nPos)) != strSource.npos)
    {
        strSource.replace(nPos, strOld.length(), strNew);
        nPos += strNew.length();
    }
 }
*/

static void print_tensor(const std::string &op, const std::string &name, const Tensor & tensor, int layer, std::string &res) {
        if(g_retentions.find(name) != g_retentions.end()) {
            return;
        }
        res += name + "," + type_str(tensor.dtype()) + "{";
        Shape shape = tensor.sizes();
        for(size_t i=0; i<shape.size(); i++ ) {
            if(i != 0) {
                res += ",";
            }
            res += std::to_string(shape[i]);
        }
        res +="}";

        res += ",{";
        int nlen = tensor.count();
        if(tensor.dtype() == CHAR8) {
            if(nlen > 32) nlen = 32;
        }else if(tensor.dtype() == INT32) {
            if(nlen > 8) nlen = 8;
        }else if(tensor.dtype() == FLOAT32) {
            if(nlen > 1) nlen = 1;
        }else if(tensor.dtype() == FLOAT64) {
            if(nlen > 1) nlen = 1;
        }else {
            if(nlen > 8) nlen = 8;
        } 

        for(int i=0; i<nlen; i++) {
            if(tensor.dtype() != CHAR8 && i!= 0 ) {
                res += ",";
            }
            switch(tensor.dtype()) {
            case CHAR8:
                //std::cout << tensor.data<char>()[i];
                res += tensor.data<char>()[i];
                break;
            case INT8:
                //std::cout << tensor.data<int8_t>()[i] << ",";
                res += std::to_string(tensor.data<int8_t>()[i]);
                break;
            case UINT8:
                //std::cout << tensor.data<uint8_t>()[i] << ",";
                res += std::to_string(tensor.data<uint8_t>()[i]);
                break;
            case INT16:
                //std::cout << tensor.data<int16_t>()[i] << ",";
                res += std::to_string(tensor.data<int16_t>()[i]);
                break;
            case UINT16:
                //std::cout << tensor.data<uint16_t>()[i] << ",";
                res += std::to_string(tensor.data<uint16_t>()[i]);
                break;
            case INT32:
                //std::cout << tensor.data<int32_t>()[i] << ",";
                if((op == name::layer::pooling2d() || op == name::layer::pooling2d_v2())  
                    && name == name::type) {
                    res += get_pooling_type_str(tensor.data<int32_t>()[i]);
                    break;
                }else if((op == name::layer::pooling2d() || op == name::layer::pooling2d_v2())
                    && name == name::padding_type) {
                    res += get_pooling_padding_type_str(tensor.data<int32_t>()[i]);
                    break;
                }else if(op == name::layer::resize2d() && name == name::type) {
                    res += get_resize2d_type_str(tensor.data<int32_t>()[i]);
                    break;
                }
                res += std::to_string(tensor.data<int32_t>()[i]);
                break;
            case UINT32:
                //std::cout << tensor.data<uint32_t>()[i] << ",";
                res += std::to_string(tensor.data<uint32_t>()[i]);
                break;
            case FLOAT32:
                res += std::to_string(tensor.data<float>()[i]);
                //std::cout << tensor.data<float>()[i] << ",";
                break;
            case FLOAT64:
                res += std::to_string(tensor.data<double>()[i]);
                //std::cout << tensor.data<double>()[i] << ",";
                break;
            case INT64:
                res += std::to_string(tensor.data<int64_t>()[i]);
                //std::cout << tensor.data<int64_t>()[i] << ",";
                break;
            case UINT64:
                res += std::to_string(tensor.data<uint64_t>()[i]);
                //std::cout << tensor.data<int64_t>()[i] << ",";
                break;
            default:
                std::cout << "unknow data type:" << type_str(tensor.dtype()) << std::endl;
                break;
            }
        }
        res += "}";
}

static void print_param(const std::string &op, const std::unordered_map<std::string, Tensor>  & params, int layer, std::string &res) {
    std::map<std::string, Tensor> tmp_params;
    std::unordered_map<std::string, Tensor>::const_iterator iter;
    for(iter=params.begin(); iter!=params.end(); ++iter) {
        tmp_params.insert(std::map<std::string, Tensor>::value_type(iter->first, iter->second));
    }

    std::string strtmp;
    for(auto iter2 = tmp_params.begin(); iter2!=tmp_params.end(); ++iter2) {
       strtmp = "";
       print_tensor(op, iter2->first, iter2->second, layer, strtmp); 
       if(strtmp.length() > 0) {
           res += "<tr><td><font color=\"red\">";
           res += convert_html(strtmp);
           res += "</font></td></tr>";
       }
    }
}



static void print_node(const Node& node, int layer, bool isinput, int nprint, bool hype = false) {
     std::string name;
     std::string childname,strtmp;
     print_param(node.bubble().op(), node.bubble().params(), 0, strtmp); 

     //ReplaceAll(name, "-", "_"); 
     name = get_node_id(node.ptr());  

     std::string op = node.bubble().op();
     std::string str2 = node.bubble().name();
     if (!hype)
        str2 = convert_html(str2);

     if(str2.length() < 1) {
         str2 = "&nbsp;&nbsp;";
     }
     g_node_stream << name  << " [label=< <table border=\"0\" ><tr><td><table border=\"0\"><tr><td>";
     g_node_stream << str2 << "</td></tr><tr><td>";
     str2 = op;
    if (!hype)
        str2 = convert_html(str2);

     if(str2.length() < 1) {
         std::cout << "operator name is empty!" << std::endl;
         exit(-1);
     }
     g_node_stream << str2 << "</td></tr></table></td>";
     
     
     if(nprint > 0 && strtmp.length() > 0 && name != "label") {
         g_node_stream << "<td><table border=\"0\">";
         g_node_stream << strtmp;
         g_node_stream << "</table></td>";
     }
    
     g_node_stream << "</tr></table>> "; 
     node_info info;
     bool bfind = get_node_info(op, info);

     if(isinput) {
         g_node_stream << ", style=filled, color=\"";
         g_node_stream << "#78d1e7" << "\", shape=\"";
         g_node_stream << "box" << "\"";
    
     }else {
         if(bfind) {
             g_node_stream << ", style=filled, color=\"";
             g_node_stream << info.color << "\", shape=\"";
             g_node_stream << info.shape << "\"";
         }
     } 

     if(!isinput && layer == 1) {
         g_node_stream << ",peripheries=\"2\", penwidth=\"2.0\"";
     }
     g_node_stream <<  "]\n";

     std::vector<char> vec(layer * 4, ' ');
     std::string str(vec.begin(), vec.end());
     std::vector<Node> inputs = node.inputs(); 
     for(size_t i=0; i<inputs.size(); i++) {

         if(nprint != 2 && inputs[i].bubble().op() == "<const>") {
             continue;
         }

         childname = get_node_id(inputs[i].ptr());
         g_direct_stream << childname << "->" ;
         g_direct_stream << name;
         g_direct_stream << "\n";

         std::string tmpname = get_bubble_name(inputs[i].bubble().name(), inputs[i].bubble().name().c_str());
         //if(g_set.find(inputs[i].bubble().name()) == g_set.end())

         if(g_set.find(tmpname) == g_set.end())
         { 
            //g_set.insert(inputs[i].bubble().name());
            g_set.insert(tmpname);
            print_node(inputs[i], ++layer, isinput, nprint); 

         }
     }
}


void usage() {
    std::cout << "usage:" << std::endl;
    std::cout << "show help: draw_net" << std::endl;;
    std::cout << "draw net without parameters and const inputs: draw_net *.tsm" << std::endl;
    std::cout << "draw net with parameters: draw_net [-p] *.tsm"<<std::endl;
    std::cout << "draw net with parameters and const inputs: draw_net [-a] *.tsm"<<std::endl;
    std::cout << "*.tsm is the TensorStack's model file" << std::endl;
}

int main(int argc, char**argv)
{
    if(argc == 1) {
        usage();
        return 0;
    }
   
    int nprint = 0;
    std::string filename;

    if(argc == 2) {
        filename = argv[1];
    }else if(argc == 3) {
        if(strcmp(argv[1],"-a") == 0) {
            filename = argv[2];
            nprint = 2;
        }else if(strcmp(argv[2], "-a") == 0) {
            filename  = argv[1];
            nprint = 2;
        }else if(strcmp(argv[1],"-p") == 0) {
            filename = argv[2];
            nprint = 1;
        }else if(strcmp(argv[2], "-p") == 0) {
            filename  = argv[1];
            nprint = 1;
        }else {
            usage();
            return 0;
        }
    }else {
        usage();
        return 0;
    }
   
    Graph g;
    ctx::bind<Graph> _graph(g);

    g_retentions.insert("#op");
    g_retentions.insert("#name");
    g_retentions.insert("#output_count");

    init_node_info();
    std::shared_ptr<Module> m = std::make_shared<Module>();

    try {
        m = Module::Load(filename);

    }catch (const Exception &e) {
        std::cout << "module Load failed:" << std::endl;
        std::cout << e.what() << std::endl;
        return -1;
    }
    std::string strtmp;

    {
        std::ostringstream oss;
        for (size_t i = 0; i < m->inputs().size(); ++i) {
            auto input = m->input(i);
            oss << "<font color=\"red\">input: </font>" << input->name();
            if (input->has("#shape")) {
                auto dtype = FLOAT32;
                if (input->has("#dtype")) dtype = DTYPE(input->get_int("#dtype"));
                oss << ", " << "<font color=\"blue\">"
                << type_str(dtype) << to_string(input->get_int_list("#shape"))
                << "</font>";
            }
            oss << "<br/>";
        }
        for (size_t i = 0; i < m->outputs().size(); ++i) {
            auto output = m->output(i);
            oss << "<font color=\"red\">output: </font>" << output->name();
            oss << "<br/>";
        }
        Node info = g.make("TensorStack", oss.str());
        print_node(info, 1, false, nprint, true);
    }

    for (auto &node : m->outputs()) {
        std::string tmpname = get_bubble_name(node.bubble().name(), node.bubble().name().c_str());
        if(g_set.find(tmpname) == g_set.end()) 
        {
            g_set.insert(tmpname);
            print_node(node, 1, false, nprint);
        }

        //g_set.insert(node.bubble().name());
        //print_node(node, 1, false, nprint);
    }
    for (auto &node : m->inputs()) {
        print_node(node, 1, true, nprint);
    }

    std::string outfilename = filename;
    size_t nfind = outfilename.rfind(".");
    if(nfind != std::string::npos) {
        strtmp = outfilename.substr(nfind);
#ifdef WIN32
        if(strnicmp(strtmp.c_str(), ".tsm", 4) == 0) {
#else
        if(strncasecmp(strtmp.c_str(), ".tsm", 4) == 0) {

#endif
            outfilename = outfilename.substr(0, nfind);
        }
    }

    int nret = 0;
    outfilename += ".dot";
    std::ofstream outfile(outfilename);
    if(outfile.is_open()) {
        outfile << "digraph tensorstack_net {\n";
        outfile << g_node_stream.str();
        outfile << g_direct_stream.str();
        outfile << "}\n";
    }else {
        nret = 1;
        std::cout << "write dot file:" << outfilename << " failed" << std::endl; 
    }
       
    outfile.close(); 
    return nret;

}
