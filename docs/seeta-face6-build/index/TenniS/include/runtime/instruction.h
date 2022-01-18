//
// Created by kier on 2018/6/28.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_H

#include <memory>
#include <global/operator_factory.h>
#include "operator.h"

namespace ts {

    class Workbench;

    class TS_DEBUG_API Instruction {
    public:
        using self = Instruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        virtual ~Instruction() = default;

        virtual void run(Workbench &workbench) = 0;

        virtual std::string str() const;

        virtual std::string repr() const;
    };

    class TS_DEBUG_API LambdaInstruction : public Instruction {
    public:
        using self = LambdaInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        using Lambda = std::function<void(Workbench &workbench)>;

        LambdaInstruction(const Lambda &lambda);

        LambdaInstruction(const Lambda &lambda, const std::string &description);

        void run(Workbench &workbench) final;

        std::string str() const final;

    private:
        Lambda m_lambda;
        std::string m_description;
    };

    class TS_DEBUG_API StackInstruction : public Instruction {
    public:
        using self = StackInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        void run(Workbench &workbench) final;

        virtual void run(Stack &stack) = 0;
    };

    /**
     * \brief push data sagment to stack
     */
    class TS_DEBUG_API DataSegmentInstruction : public Instruction {
    public:
        using self = DataSegmentInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        explicit DataSegmentInstruction(int data_index);

        int data_index() const { return m_data_index; }

        void run(Workbench &workbench) final;

        std::string str() const final;

    private:
        int m_data_index;
    };

    class TS_DEBUG_API OperatorInstruction : public Instruction {
    public:
        using self = OperatorInstruction;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer
        using supper = Instruction;

        explicit OperatorInstruction(const Operator::shared &func, int nargs, int nresults);
        explicit OperatorInstruction(const Operator::shared &func, int nargs, int nresults, const std::string &description);

        void run(Workbench &workbench) final ;

        std::string str() const final;

        void bind_creator(OperatorCreator::function creator);

        shared clone() const;

        Operator::shared op() const { return m_func; }

    private:
        Operator::shared m_func = nullptr;
        int m_nargs = 0;
        int m_nresults = 0;
        std::string m_description;

        OperatorCreator::function m_creator = nullptr;
    };

}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_H
