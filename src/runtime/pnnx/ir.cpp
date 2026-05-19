#include "runtime/pnnx/ir.h"
#include "runtime/pnnx/store_zip.hpp"

#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>

namespace pnnx {

static bool type_is_integer(int type) {
    if (type == 1) return false;
    if (type == 2) return false;
    if (type == 3) return false;
    if (type == 4) return true;
    if (type == 5) return true;
    if (type == 6) return true;
    if (type == 7) return true;
    if (type == 8) return true;
    if (type == 9) return true;
    if (type == 10) return false;
    if (type == 11) return false;
    if (type == 12) return false;
    
    return false;
}

static const char* type_to_string(int type) {
    if (type == 1) return "f32";
    if (type == 2) return "f64";
    if (type == 3) return "f16";
    if (type == 4) return "i32";
    if (type == 5) return "i64";
    if (type == 6) return "i16";
    if (type == 7) return "i8";
    if (type == 8) return "u8";
    if (type == 9) return "bool";
    if (type == 10) return "cp64";
    if (type == 11) return "cp128";
    if (type == 12) return "cp32";

    return "null";
}

static const char* type_to_numpy_string(int type) {
    if (type == 1) return "float32";
    if (type == 2) return "float64";
    if (type == 3) return "float16";
    if (type == 4) return "int32";
    if (type == 5) return "int64";
    if (type == 6) return "int16";
    if (type == 7) return "int8";
    if (type == 8) return "uint8";
    if (type == 9) return "bool8";
    if (type == 10) return "csingle";
    if (type == 11) return "cdouble";
    if (type == 12) return "chalf";

    return "null";
}

static size_t type_to_elemsize(int type) {
    if (type == 1) return 4;
    if (type == 2) return 8;
    if (type == 3) return 2;
    if (type == 4) return 4;
    if (type == 5) return 8;
    if (type == 6) return 2;
    if (type == 7) return 1;
    if (type == 8) return 1;
    if (type == 9) return 1;
    if (type == 10) return 8;
    if (type == 11) return 16;
    if (type == 12) return 4;

    return 0; // null
}

static int string_to_type(const char* s) {
    if (strcmp(s, "f32") == 0) return 1;
    if (strcmp(s, "f64") == 0) return 2;
    if (strcmp(s, "f16") == 0) return 3;
    if (strcmp(s, "i32") == 0) return 4;
    if (strcmp(s, "i64") == 0) return 5;
    if (strcmp(s, "i16") == 0) return 6;
    if (strcmp(s, "i8") == 0) return 7;
    if (strcmp(s, "u8") == 0) return 8;
    if (strcmp(s, "bool") == 0) return 9;
    if (strcmp(s, "cp64") == 0) return 10;
    if (strcmp(s, "cp128") == 0) return 11;
    if (strcmp(s, "cp32") == 0) return 12;

    return 0; // null
}

bool operator==(const Parameter& lhs, const Parameter& rhs) {
    if (lhs.type != rhs.type) {
        return false;
    }

    if (lhs.type == 0) {
        return true;
    }

    if (lhs.type == 1 && lhs.b == rhs.b) {
        return true;
    }

    if (lhs.type == 2 && lhs.i == rhs.i) {
        return true;
    }

    if (lhs.type == 3 && lhs.f == rhs.f) {
        return true;
    }

    if (lhs.type == 4 && lhs.s == rhs.s) {
        return true;
    }

    if (lhs.type == 5 && lhs.ai == rhs.ai) {
        return true;
    }

    if (lhs.type == 6 && lhs.af == rhs.af) {
        return true;
    }

    if (lhs.type == 7 && lhs.as == rhs.as) {
        return true;
    }

    return false;
}

Attribute::Attribute(const std::initializer_list<int>& _shape, const std::vector<float>& t) {
    type = 1;
    shape = _shape;

    if (shape.size() > 0) {
        int size = shape[0];
        for (size_t i = 1; i < shape.size(); i++) {
            size *= shape[i];
        }

        data.resize(size * type_to_elemsize(type));
        memcpy((void*)data.data(), (const void*)t.data(), data.size());
    }
}

bool operator==(const Attribute& lhs, const Attribute& rhs) {
    if (lhs.type != rhs.type) {
        return false;
    }

    if (lhs.type == 0) {
        return true;
    }

    if (lhs.shape != rhs.shape) {
        return false;
    }

    if (lhs.data != rhs.data) {
        return false;
    }

    return true;
}

Attribute operator+(const Attribute& a, const Attribute& b) {
    Attribute c;

    if (a.type != b.type) {
        fprintf(stderr, "concat attribute type mismatch\n");
        return c;
    }

    if (a.shape.size() != b.shape.size()) {
        fprintf(stderr, "concat attribute shape rank mismatch\n");
        return c;
    }

    for (int i = 1; i < (int)a.shape.size(); i++) {
        if (a.shape[i] != b.shape[i]) {
            fprintf(stderr, "concat attribute shape mismatch\n");
            return c;
        }
    }

    c.type = a.type;
    c.shape = a.shape;
    c.shape[0] += b.shape[0]; // concat the first dim

    c.data.resize(a.data.size() + b.data.size());
    memcpy(c.data.data(), a.data.data(), a.data.size());
    memcpy(c.data.data() + a.data.size(), b.data.data(), b.data.size());

    return c;
}

Parameter Parameter::parse_from_string(const std::string& value) {
    Parameter p;
    p.type = 0;

    if (value == "None" || value == "()" || value == "[]") {
        return p;
    }

    if (value == "True" || value == "False") {
        // bool
        p.type = 1;
        p.b = value == "True";
        return p;
    }

    if (value[0] == '(' || value[0] == '[') {
        // list
        std::string lc = value.substr(1, value.size() - 2);
        std::istringstream lcss(lc);

        while (!lcss.eof()) {
            std::string elem;
            std::getline(lcss, elem, ',');

            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
                // string
                p.type = 7;
                p.as.push_back(elem);
            } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                // float
                p.type = 6;
                p.af.push_back(std::stof(elem));
            } else {
                // integer
                p.type = 5;
                p.ai.push_back(std::stoi(elem));
            }
        }

        return p;
    }

    if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) || (value[0] == '-' && (value[1] < '0' || value[1] > '9'))) {
        // string
        p.type = 4;
        p.s = value;
        return p;
    }

    if (value.find('.') != std::string::npos || value.find('e') != std::string::npos) {
        // float
        p.type = 3;
        p.f = std::stof(value);
        return p;
    }

    // integer
    p.type = 2;
    p.i = std::stoi(value);
    return p;
}

Graph::Graph() {
}

Graph::~Graph() {
    for (auto x : operators) {
        delete x;
    }

    for (auto x : operands) {
        delete x;
    }

    operators.clear();
    operands.clear();
}

Graph::Graph(const Graph&) {
}

Graph& Graph::operator=(const Graph&) {
    return *this;
}

static void load_parameter(Operator* op, const std::string& key, const std::string& value) {
    op->params[key] = Parameter::parse_from_string(value);
}

static void load_input_key(Operator* op, const std::string& key, const std::string& value) {
    op->inputnames.resize(op->inputs.size());

    for (size_t i = 0; i < op->inputs.size(); i++) {
        const Operand* oprand = op->inputs[i];
        if (oprand->name == value) {
            op->inputnames[i] = key;
            break;
        }
    }
}

static void load_shape(Operator* op, const std::string& key, const std::string& value) {
    Operand* operand = 0;

    for (auto input : op->inputs) {
        if (input->name == key) {
            operand = input;
            break;
        }
    }

    if (!operand) {
        for (auto input : op->outputs) {
            if (input->name == key)
            {
                operand = input;
                break;
            }
        }
    }

    if (!operand) {
        fprintf(stderr, "no such operand %s for operator %s\n", key.c_str(), op->name.c_str());
        return;
    }

    // type
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    operand->type = string_to_type(typestr.c_str());

    // shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);

    operand->shape.clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');

        if (elem == "?") {
            operand->shape.push_back(-1);
        } else {
            int i = std::stoi(elem);
            operand->shape.push_back(i);
        }
    }
}

static void load_attribute(Operator* op, const std::string& key, const std::string& value, StoreZipReader& szr) {
    Attribute& attr = op->attrs[key];

    // type
    std::string typestr = value.substr(value.find_last_of(')') + 1);
    attr.type = string_to_type(typestr.c_str());

    if (attr.type == 0) {
        return;
    }

    // shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);

    attr.shape.clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');

        int i = std::stoi(elem);
        attr.shape.push_back(i);
    }

    if (attr.shape.empty()) {
        return;
    }

    // weight_data
    size_t size = 1;
    for (int i : attr.shape) {
        size *= i;
    }

    size_t bytesize = size * type_to_elemsize(attr.type);

    std::string filename = op->name + "." + key;

    size_t filesize = szr.get_file_size(filename);

    if (filesize == 0) {
        // no such file
        return;
    }

    if (filesize != bytesize) {
        fprintf(stderr, "file size not match expect %lu but got %lu\n", bytesize, filesize);
    }

    attr.data.resize(bytesize);
    szr.read_file(filename, (char*)attr.data.data());
}

int Graph::load(const std::string& parampath, const std::string& binpath) {
    std::ifstream is(parampath, std::ios::in | std::ios::binary);
    if (!is.good()) {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    StoreZipReader szr;
    if (szr.open(binpath) != 0) {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    int magic = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> magic;
    }

    int operator_count = 0;
    int operand_count = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> operator_count >> operand_count;
    }

    for (int i = 0; i < operator_count; i++) {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int input_count = 0;
        int output_count = 0;

        iss >> type >> name >> input_count >> output_count;

        Operator* op = new_operator(type, name);

        for (int j = 0; j < input_count; j++) {
            std::string operand_name;
            iss >> operand_name;

            Operand *r = get_operand(operand_name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < output_count; j++) {
            std::string operand_name;
            iss >> operand_name;

            Operand *r = new_operand(operand_name);
            r->producer = op;
            op->outputs.push_back(r);
        }

        // key=value
        while (!iss.eof()) {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);

            if (key[0] == '@') {
                // attribute
                // e.g. @weight=(64,3,3,3)
                load_attribute(op, key.substr(1), value, szr);
            } else if (key[0] == '$') {
                // operand input key
                // e.g. $weight=feat
                load_input_key(op, key.substr(1), value);
            } else if (key[0] == '#') {
                // operand shape
                // e.g. #feat=(1,3,15,15)
                load_shape(op, key.substr(1), value);
            } else {
                // parameter
                // e.g. stride=2
                load_parameter(op, key, value);
            }
        }
    }

    return 0;
}

int Graph::save(const std::string& parampath, const std::string& binpath) {
    FILE* paramfp = fopen(parampath.c_str(), "wb");
    if (!paramfp) {
        fprintf(stderr, "fopen %s failed\n", parampath.c_str());
        return -1;
    }

    StoreZipWriter szw;
    if (szw.open(binpath) != 0) {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    // magic
    fprintf(paramfp, "7767517\n");

    // op count and oprand count
    fprintf(paramfp, "%d %d\n", (int)operators.size(), (int)operands.size());

    for (const Operator *op : operators) {
        fprintf(paramfp, "%-24s %-24s %d %d", op->type.c_str(), op->name.c_str(), (int)op->inputs.size(), (int)op->outputs.size());

        for (const Operand *oprand : op->inputs) {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const Operand *oprand : op->outputs) {
            fprintf(paramfp, " %s", oprand->name.c_str());
        }

        for (const auto& it : op->params) {
            fprintf(paramfp, " %s=", it.first.c_str());

            const Parameter& param = it.second;
            if (param.type == 0) {
                fprintf(paramfp, "None");
            }
            if (param.type == 1) {
                if (param.b) {
                    fprintf(paramfp, "True");
                } else {
                    fprintf(paramfp, "False");
                }
            }
            if (param.type == 2) {
                fprintf(paramfp, "%d", param.i);
            }
            if (param.type == 3) {
                fprintf(paramfp, "%e", param.f);
            }
            if (param.type == 4) {
                fprintf(paramfp, "%s", param.s.c_str());
            }
            if (param.type == 5) {
                fprintf(paramfp, "(");
                for (size_t i = 0; i < param.ai.size(); i++) {
                    fprintf(paramfp, "%d", param.ai[i]);
                    if (i + 1 != param.ai.size())
                        fprintf(paramfp, ",");
                }
                fprintf(paramfp, ")");
            }
            if (param.type == 6) {
                fprintf(paramfp, "(");
                for (size_t i = 0; i < param.af.size(); i++) {
                    fprintf(paramfp, "%e", param.af[i]);
                    if (i + 1 != param.af.size())
                        fprintf(paramfp, ",");
                }
                fprintf(paramfp, ")");
            }
            if (param.type == 7) {
                fprintf(paramfp, "(");
                for (size_t i = 0; i < param.as.size(); i++) {
                    fprintf(paramfp, "%s", param.as[i].c_str());
                    if (i + 1 != param.as.size())
                        fprintf(paramfp, ",");
                }
                fprintf(paramfp, ")");
            }
        }

        for (const auto& it : op->attrs) {
            fprintf(paramfp, " @%s=", it.first.c_str());

            const Attribute& attr = it.second;
            fprintf(paramfp, "(");
            for (int i = 0; i < (int)attr.shape.size() - 1; i++) {
                fprintf(paramfp, "%d,", attr.shape[i]);
            }
            if (attr.shape.size() > 0) {
                fprintf(paramfp, "%d", attr.shape[attr.shape.size() - 1]);
            }
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(attr.type));

            std::string filename = op->name + "." + it.first;
            szw.write_file(filename, attr.data.data(), attr.data.size());
        }

        if (op->inputnames.size() == op->inputs.size()) {
            for (size_t i = 0; i < op->inputs.size(); i++) {
                if (op->inputnames[i].empty()) {
                    continue;
                }

                const Operand *oprand = op->inputs[i];
                fprintf(paramfp, " $%s=%s", op->inputnames[i].c_str(), oprand->name.c_str());
            }
        }

        for (const Operand *oprand : op->inputs) {
            if (oprand->shape.empty()) {
                continue;
            }

            fprintf(paramfp, " #%s=", oprand->name.c_str());

            fprintf(paramfp, "(");
            for (int i = 0; i < (int)oprand->shape.size() - 1; i++) {
                if (oprand->shape[i] == -1) {
                    fprintf(paramfp, "?,");
                } else {
                    fprintf(paramfp, "%d,", oprand->shape[i]);
                }
            }
            if (oprand->shape.size() > 0) {
                if (oprand->shape[oprand->shape.size() - 1] == -1) {
                    fprintf(paramfp, "?");
                } else {
                    fprintf(paramfp, "%d", oprand->shape[oprand->shape.size() - 1]);
                }
            }
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(oprand->type));
        }

        for (const Operand *oprand : op->outputs) {
            if (oprand->shape.empty()) {
                continue;
            }

            fprintf(paramfp, " #%s=", oprand->name.c_str());

            fprintf(paramfp, "(");
            for (int i = 0; i < (int)oprand->shape.size() - 1; i++) {
                if (oprand->shape[i] == -1)
                    fprintf(paramfp, "?,");
                else
                    fprintf(paramfp, "%d,", oprand->shape[i]);
            }
            if (oprand->shape.size() > 0) {
                if (oprand->shape[oprand->shape.size() - 1] == -1) {
                    fprintf(paramfp, "?");
                } else {
                    fprintf(paramfp, "%d", oprand->shape[oprand->shape.size() - 1]);
                }
            }
            fprintf(paramfp, ")");

            fprintf(paramfp, type_to_string(oprand->type));
        }

        fprintf(paramfp, "\n");
    }

    fclose(paramfp);

    return 0;
}

static std::string sanitize_identifier(const std::string& ident) {
    std::string sanitized_ident = ident;
    for (size_t i = 0; i < sanitized_ident.size(); i++) {
        if (sanitized_ident[i] == '.' || sanitized_ident[i] == ':') {
            sanitized_ident[i] = '_';
        }
    }

    return sanitized_ident;
}

int Graph::parse(const std::string& param) {
    std::istringstream is(param);
    if (!is.good()) {
        fprintf(stderr, "open failed\n");
        return -1;
    }

    int magic = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> magic;
    }

    int operator_count = 0;
    int operand_count = 0;
    {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        iss >> operator_count >> operand_count;
    }

    for (int i = 0; i < operator_count; i++) {
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int input_count = 0;
        int output_count = 0;

        iss >> type >> name >> input_count >> output_count;

        Operator *op = new_operator(type, name);

        for (int j = 0; j < input_count; j++) {
            std::string operand_name;
            iss >> operand_name;

            Operand *r = get_operand(operand_name);
            r->consumers.push_back(op);
            op->inputs.push_back(r);
        }

        for (int j = 0; j < output_count; j++) {
            std::string operand_name;
            iss >> operand_name;

            Operand *r = new_operand(operand_name);
            r->producer = op;
            op->outputs.push_back(r);
        }

        // key=value
        while (!iss.eof()) {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);

            if (key[0] == '@') {
                // attribute
                load_attribute(op, key.substr(1), value, szr);
            } else if (key[0] == '$') {
                // operand input key
                load_input_key(op, key.substr(1), value);
            } else if (key[0] == '#') {
                // operand shape
                load_shape(op, key.substr(1), value);
            } else {
                // parameter
                load_parameter(op, key, value);
            }
        }
    }

    return 0;
}

void Operand::remove_consumer(const Operator* c) {
    auto it = std::find(consumers.begin(), consumers.end(), c);
    consumers.erase(it);
}

Operator *Graph::new_operator(const std::string& type, const std::string& name) {
    Operator *op = new Operator;
    op->type = type;
    op->name = name;
    operators.push_back(op);
    return op;
}

Operator *Graph::new_operator_before(const std::string& type, const std::string& name, const Operator* cur) {
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    operators.insert(std::find(operators.begin(), operators.end(), cur), op);
    return op;
}

Operator *Graph::new_operator_after(const std::string& type, const std::string& name, const Operator* cur) {
    Operator* op = new Operator;
    op->type = type;
    op->name = name;
    operators.insert(std::find(operators.begin(), operators.end(), cur) + 1, op);
    return op;
}

Operand* Graph::new_operand(const std::string& name) {
    Operand *r = new Operand;
    r->name = name;
    operands.push_back(r);
    return r;
}

Operand* Graph::get_operand(const std::string& name) {
    for (Operand* r : operands) {
        if (r->name == name) {
            return r;
        }
    }

    return 0;
}

const Operand* Graph::get_operand(const std::string& name) const {
    for (const Operand *r : operands) {
        if (r->name == name) {
            return r;
        }
    }

    return 0;
}

} // namespace pnnx
