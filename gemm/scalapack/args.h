#pragma once

// usage:
// int N, its;
// arg( "N", &N );
// arg( "its", &its );
// args( argc, argv );

class Arg {
public:
    virtual void assign( const char *argvalue ) = 0;
    virtual void print( ostream::ostream &os ) const = 0;
};
ostream::ostream &operator<<( ostream::ostream &os, const Arg &arg ) {
    arg.print( os );
    return os;
}

class IntArg : public Arg {
public:
    int *argptr;
    IntArg( int *_argptr ) : argptr(_argptr ) {
    }
    void assign( const char *argvalue ) {
        *argptr = atoi( argvalue );
    }
    void print( ostream::ostream &os ) const {
        os << (*argptr );
    }
};

vector<string> argnames;
vector<Arg *> argptrs;

void arg_usage(string cmd) {
    cout << "Usage: " << cmd;
    for( int i = 0; i < argnames.size(); i++ ) {
        cout << " [" << argnames[i] << "]";
    }
    cout << endl;
    exit(1);
}

void arg( string name, int *p_value ) {
    argnames.push_back(name);
    argptrs.push_back( new IntArg( p_value ) );
}

void args( int argc, char *argv[] ) {
    if( argc - 1 != argnames.size() ) {
        arg_usage(argv[0]);
    }
    for( int i = 0; i < argnames.size(); i++ ) {
        argptrs[i]->assign( argv[i+1] );
        cout << argnames[i] << ": " << (*argptrs[i]) << endl;
    }
}

class Args {
public:
    int argc;
    char **argv;
    Args( int _argc, char *_argv[] ) : argc(_argc), argv(_argv) {
    }
    void go() {
        args( argc, argv );
    }
    Args &_( string name, int *pvalue ) {
        ::arg( name, pvalue );
        return *this;
    }
    Args &arg( string name, int *pvalue ) {
        ::arg( name, pvalue );
        return *this;
    }
};