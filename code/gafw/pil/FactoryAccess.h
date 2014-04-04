/* 
 * File:   FactoryAccess.h
 * Author: daniel
 *
 * Created on 05 February 2014, 15:46
 */

#ifndef __FACTORYACCESS_H__
#define	__FACTORYACCESS_H__

#include "Factory.h"




namespace GAFW
{
    
    class FactoryAccess{
    private:
        //Set here the copy constructor
        FactoryAccess(const FactoryAccess& orig){};
    protected:
        FactoryAccess()
        {
            throw Bug("This function only exists for convenience. It is a bug to call it");
        };
        Identity *thisId;
    public:
        const class Factory *factory;
        inline FactoryAccess(Factory *factory);
        inline void init();

        inline virtual ~FactoryAccess();
        inline Array *requestMyArray(std::string nickname, ArrayDimensions dim=ArrayDimensions(),DataTypeBase type=DataType<void>());
        inline ArrayOperator *requestMyOperator(std::string nickname, std::string OperatorType);
        inline ArrayOperator *requestMyOperator(std::string nickname, std::string OperatorType, int noOfInputs, const char *  inputNick,...);
        inline ArrayOperator *requestMyOperator(std::string nickname, std::string OperatorType, const char * inputNick1, const char *  inputNick2=NULL, const char *  inputNick3=NULL,const char *  inputNick4=NULL,const char *  inputNick5=NULL,const char * inputNick6=NULL,const char *  inputNick7=NULL);

        inline ProxyResult *requestMyProxyResult(std::string nickname,Module *mod=NULL,Result * bind=NULL);
        inline Array * getMyArray(std::string nickname);
        inline ArrayOperator * getMyOperator(std::string nickname);
        inline ProxyResult * getMyProxyResult(std::string nickname);
        inline Result *getMyResult(std::string nickname);
        inline Identity *getMyParent();
        inline Factory *getFactory();
    };
    inline FactoryAccess::FactoryAccess(Factory *factory):factory(factory),thisId(dynamic_cast<Identity *>(this))
    {
        
    }
    inline void FactoryAccess::init()
    {
        thisId=dynamic_cast<Identity *>(this);
        if (thisId==NULL) throw Bug("The class which FactoryAccess is expected to inherit the Identity class too.");
     }

    inline FactoryAccess::~FactoryAccess()
    {
    }
    inline Array *FactoryAccess::requestMyArray(std::string nickname, ArrayDimensions dim,DataTypeBase type)
    {
        return this->getFactory()->requestMyArray(*thisId,nickname,dim,type);
    }
    inline ArrayOperator *FactoryAccess::requestMyOperator(std::string nickname, std::string OperatorType)
    {
        return this->getFactory()->requestMyOperator(*thisId,nickname,OperatorType);
    }
    inline ArrayOperator *FactoryAccess::requestMyOperator(std::string nickname, std::string OperatorType, int noOfInputs, const char * inputObjectName1,...)
    {
        ArrayOperator *a;
        va_list args;
        va_start(args,noOfInputs);
        a=this->getFactory()->requestMyOperator(*thisId,nickname,OperatorType,noOfInputs,inputObjectName1,args);
        va_end(args);
        return a;
    }
    inline ArrayOperator *FactoryAccess::requestMyOperator(std::string nickname, std::string OperatorType, const char * inputNick1, const char *  inputNick2, const char *  inputNick3,const char *  inputNick4,const char *  inputNick5,const char * inputNick6,const char *  inputNick7)
    {
        return this->getFactory()->requestMyOperator(*thisId,nickname,OperatorType,inputNick1,inputNick2,inputNick3,inputNick4,inputNick5,inputNick6,inputNick7);
    }
    inline ProxyResult *FactoryAccess::requestMyProxyResult(std::string nickname,Module *mod,Result * bind)
    {
        return this->getFactory()->requestMyProxyResult(*thisId,nickname,mod,bind);
    }
    inline Array * FactoryAccess::getMyArray(std::string nickname)
    {
        return this->getFactory()->getMyArray(*thisId,nickname);
    }
    inline ArrayOperator * FactoryAccess::getMyOperator(std::string nickname)
    {
        return this->getFactory()->getMyOperator(*thisId,nickname);
    }
    inline ProxyResult * FactoryAccess::getMyProxyResult(std::string nickname)
    {
        return this->getFactory()->getMyProxyResult(*thisId,nickname);
    }
    inline Result *FactoryAccess::getMyResult(std::string nickname)
    {
        return this->getFactory()->getMyResult(*thisId,nickname);
    }
    inline Identity *FactoryAccess::getMyParent()
    {
        return this->getFactory()->getMyParent(*thisId);
    }
    inline Factory *FactoryAccess::getFactory()
    {
         return const_cast<class GAFW::Factory *>(this->factory);
    }
}
    


#endif	/* FACTORYACCESS_H */

