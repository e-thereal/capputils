#pragma once

#include <exception>
#include <string>

namespace tbblas {

class assert_exception : public std::exception {
private:
    std::string reason, file, message;
    int line;
    
public:
    assert_exception(const std::string& reason, const std::string& file, int line) : reason(reason), file(file), line(line) { 
        message = "Assertion failed: " + reason + " at " + file + ", line " + std::to_string(line);
    }

    virtual ~assert_exception() throw() { }

    virtual const char* what() const throw() {
        return message.c_str();
    }
};

}

#define tbblas_assert(_Expression) (void)( (!!(_Expression)) || (throw tbblas::assert_exception(#_Expression, __FILE__, __LINE__), 0) )
