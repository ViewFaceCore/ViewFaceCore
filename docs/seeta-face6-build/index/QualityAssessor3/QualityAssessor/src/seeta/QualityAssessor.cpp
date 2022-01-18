//
// Created by kier on 19-4-24.
//

#include "seeta/QualityAssessor.h"

#include "seeta/QualityOfBrightness.h"
#include "seeta/QualityOfClarity.h"
#include "seeta/QualityOfIntegrity.h"
#include "seeta/QualityOfPose.h"
#include "seeta/QualityOfResolution.h"

#include <orz/utils/log.h>
#include <cfloat>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

namespace seeta {
    namespace v3{
		typedef std::pair<QualityAttribute, QualityRule*> rule_type;
		typedef std::pair<QualityAttribute, bool> rule_strict_type;
		typedef std::pair<QualityAttribute, bool> rule_enable_type;
		typedef std::pair<QualityAttribute, QualityResult> quality_result_type;

		class QualityAssessor::Implement {
		public:
			explicit Implement();
			~Implement();

			void add_rule(int32_t attr, bool must_high=false);

			void add_rule(int32_t attr, const SeetaModelSetting &model, bool must_high = false);

			void add_rule(int32_t attr, QualityRule *rule, bool must_high = false);

			bool remove_rule(int32_t attr);

			bool has_rule(int32_t attr);

			void set_medium_limit(int32_t limit);

			void feed(
				const SeetaImageData &image,
				const SeetaRect &face,
				const SeetaPointF *points,
				int32_t N);

			QualityResult query(int32_t attr);

			void disable(int32_t attr);

			void enable(int32_t attr);

		public:
                        std::vector<QualityResultEx> last_results_ex;
			std::vector<rule_type> rules;
			std::vector<rule_strict_type> rules_strict_level;//check if must high
			std::vector<rule_enable_type> rules_enable_list;//check if attribute is enabled
			std::vector<quality_result_type> last_results;
			int32_t medium_limit = 0;//How many MEDIUM can be accepted in all attributes, default is 0
		};

		QualityAssessor::Implement::Implement()
		{

		}

		void rule_deleter(rule_type &rule)
		{
			if(rule.second) delete rule.second;
			rule.second = nullptr;
		}
		QualityAssessor::Implement::~Implement()
		{
			for_each(rules.begin(), rules.end(), rule_deleter);
                         last_results_ex.clear();
		}

		 std::vector<rule_type>::iterator
			check_rule_if_existed(int32_t attr,
			std::vector<rule_type> &rules)
		{
			 std::vector<rule_type>::iterator iter = std::find_if(rules.begin(), rules.end(), 
				[attr](const rule_type & rule) {
				return rule.first == attr;
			});

			return iter;
		}
		void QualityAssessor::Implement::add_rule(int32_t attr, bool must_high)
		{
			auto iter = check_rule_if_existed(attr, rules);
			if (iter != rules.end())
			{
				orz::Log(orz::INFO) << "attr is added already.\n";
				return;
			}

			QualityRule* rule;
			switch (attr)
			{
			case BRIGHTNESS:
				rule = new QualityOfBrightness();
				break;
			case CLARITY:
				rule = new QualityOfClarity();
				break;
			case INTEGRITY:
				rule = new QualityOfIntegrity();
				break;
			case POSE:
				rule = new QualityOfPose();
				break;
			case RESOLUTION:
				rule = new QualityOfResolution();
				break;
			default:
				orz::Log(orz::INFO) << "input attr is not permmitted to add.\n";
				return;
			}

			rules.emplace_back(rule_type(QualityAttribute(attr), rule));
			rules_strict_level.emplace_back(rule_strict_type(QualityAttribute(attr), must_high));
			rules_enable_list.emplace_back(rule_enable_type(QualityAttribute(attr), true));
		}

		void QualityAssessor::Implement::add_rule(int32_t attr, const SeetaModelSetting &model, bool must_high)
		{
			orz::Log(orz::INFO)<< "this add_rule method is not valid at present.\n";
		}

		void QualityAssessor::Implement::add_rule(int32_t attr, QualityRule *rule, bool must_high)
		{
			auto iter = check_rule_if_existed(attr, rules);
			if (iter != rules.end())
			{
				orz::Log(orz::INFO) << "attr is added already.\n";
				return;
			}

			rules.emplace_back(rule_type(QualityAttribute(attr), rule));
			rules_strict_level.emplace_back(rule_strict_type(QualityAttribute(attr), must_high));
			rules_enable_list.emplace_back(rule_enable_type(QualityAttribute(attr), true));
		}

		bool QualityAssessor::Implement::remove_rule(int32_t attr)
		{
			bool result = true;

			rules.erase(std::remove_if(rules.begin(),
				rules.end(), 
				[attr](const rule_type& rule) {
				return rule.first == attr;
			}), 
			rules.end());

			rules_strict_level.erase(std::remove_if(rules_strict_level.begin(),
				rules_strict_level.end(),
				[attr](const rule_strict_type& rule_strict) {
				return rule_strict.first == attr;
			}),
			rules_strict_level.end());

			rules_enable_list.erase(std::remove_if(rules_enable_list.begin(),
				rules_enable_list.end(),
				[attr](const rule_enable_type & rule_enable) {
				return rule_enable.first == attr;
			}),
				rules_enable_list.end());

			return result;
		}

		bool QualityAssessor::Implement::has_rule(int32_t attr)
		{
			auto iter = check_rule_if_existed(attr, rules);
			if (iter != rules.end()) return true;

			return false;
		}

		void QualityAssessor::Implement::set_medium_limit(int32_t limit)
		{
			medium_limit = limit;
		}

		void QualityAssessor::Implement::feed(
			const SeetaImageData &image,
			const SeetaRect &face,
			const SeetaPointF *points,
			int32_t N)
		{
			last_results.clear();

			for (size_t i = 0; i < rules.size(); ++i)
			{
				if (rules_enable_list[i].second)
				{//enabled
					QualityResult result = rules[i].second->check(
						image, face, points, N);

					last_results.emplace_back(quality_result_type(rules[i].first, result));
				}

			}
		}

		QualityResult QualityAssessor::Implement::query(int32_t attr)
		{
			std::vector<quality_result_type>::iterator iter = std::find_if(last_results.begin(), last_results.end(),
				[attr](const quality_result_type & result) {
				return result.first == attr;
			});

			if (iter == last_results.end())
			{
				orz::Log(orz::ERROR) << "you must add attr before query it.\n" << orz::crash;
				return QualityResult(LOW, 0.0);
			}

			std::vector<rule_enable_type>::iterator enable_iter = std::find_if(rules_enable_list.begin(), rules_enable_list.end(),
				[attr](const rule_enable_type & rule_enable) {
				return rule_enable.first == attr;
			});

			if (!enable_iter->second) {
				orz::Log(orz::ERROR) << "you must enable attr before query it.\n" << orz::crash;
				return QualityResult(LOW, 0.0);
			}

			return iter->second;
		}

		void QualityAssessor::Implement::disable(int32_t attr)
		{
			auto iter = std::find_if(rules_enable_list.begin(), rules_enable_list.end(),
				[attr](const rule_enable_type & rule_enable_) {
				return rule_enable_.first == attr;
			});

			if (iter == rules_enable_list.end())
			{
				orz::Log(orz::ERROR) << "attr is not added before.\n" << orz::crash;
			}

			rules_enable_list[attr].second = false;
		}

		void QualityAssessor::Implement::enable(int32_t attr)
		{
			auto iter = std::find_if(rules_enable_list.begin(), rules_enable_list.end(),
				[attr](const rule_enable_type & rule_enable_) {
				return rule_enable_.first == attr;
			});

			if (iter == rules_enable_list.end())
			{
				orz::Log(orz::ERROR) << "attr index has exceeded limit.\n" << orz::crash;
			}

			rules_enable_list[attr].second = true;
		}

		QualityAssessor::QualityAssessor()
		{
			m_impl = new Implement();
		}

		QualityAssessor::~QualityAssessor()
		{
			delete m_impl;
		}

		void QualityAssessor::add_rule(int32_t attr, bool must_high)
		{
			 m_impl->add_rule(attr, must_high);
		}

		void QualityAssessor::add_rule(int32_t attr, const SeetaModelSetting &model, bool must_high)
		{
			m_impl->add_rule(attr, model, must_high);
		}

		void QualityAssessor::add_rule(int32_t attr, QualityRule *rule, bool must_high)
		{
			m_impl->add_rule(attr, rule, must_high);
		}

		void QualityAssessor::remove_rule(int32_t attr) 
		{
			m_impl->remove_rule(attr);
		}

		bool QualityAssessor::has_rule(int32_t attr)
		{
			return m_impl->has_rule(attr);
		}

		void QualityAssessor::set_medium_limit(int32_t limit)
		{
			m_impl->set_medium_limit(limit);
		}

		void QualityAssessor::feed(
			const SeetaImageData &image,
			const SeetaRect &face,
			const SeetaPointF* points,
			int32_t N)
		{
			m_impl->feed(image, face, points, N);
		}

		QualityResult QualityAssessor::query(int32_t attr) const
		{
			return m_impl->query(attr);
		}

		void QualityAssessor::disable(int32_t attr)
		{
			m_impl->disable(attr);
		}

		void QualityAssessor::enable(int32_t attr)
		{
			m_impl->enable(attr);
		}

		bool QualityAssessor::evaluate(const SeetaImageData &image,
			const SeetaRect &face,
			const SeetaPointF *points,
			int32_t N)
		{
			bool ret = true;

			m_impl->feed(image, face ,points, N);

			int32_t medium_num = 0;
			int32_t medium_limit = m_impl->medium_limit;

			for (size_t i = 0; i < m_impl->last_results.size();++i)
			{
				auto& result = m_impl->last_results[i].second;
				if (result.level <= LOW) return false;

				if (result.level == MEDIUM)
				{
					++medium_num;
					if (m_impl->rules_strict_level[i].second)
					{//if must high
						return false;
					}
				}
			}

			if (medium_num > medium_limit) return false;

			return ret;
		}

		bool QualityAssessor::evaluate(const SeetaImageData &image,
			const SeetaRect &face,
			const SeetaPointF *points,
			int32_t N, QualityResultExArray &result)
		{
			bool ret = true;

			m_impl->feed(image, face ,points, N);

			int32_t medium_num = 0;
			int32_t medium_limit = m_impl->medium_limit;

                        m_impl->last_results_ex.clear();
			for (size_t i = 0; i < m_impl->last_results.size();++i)
			{
				auto& res = m_impl->last_results[i].second;
				if (res.level <= LOW)
                                {
                                    QualityResultEx qr;
                                    qr.attr = int(m_impl->last_results[i].first);
                                    qr.level =  res.level;
                                    qr.score = res.score;
                                    m_impl->last_results_ex.push_back(qr);
                                    result.size = 1;
                                    result.data = m_impl->last_results_ex.data();
                                    return false;
                                }

				if (res.level == MEDIUM)
				{
					++medium_num;

                                        QualityResultEx qr;
                                        qr.attr = int(m_impl->last_results[i].first);
                                        qr.level =  res.level;
                                        qr.score = res.score;

					if (m_impl->rules_strict_level[i].second)
					{//if must high
                                                
                                            m_impl->last_results_ex.clear();
                                            m_impl->last_results_ex.push_back(qr);

                                            result.size = 1;
                                            result.data = m_impl->last_results_ex.data();
					    return false;
					}else
                                        {
                                            m_impl->last_results_ex.push_back(qr);
                                        }
				}
			}

			if (medium_num > medium_limit) 
                        {
                            result.size = m_impl->last_results_ex.size();
                            result.data = m_impl->last_results_ex.data();
                            return false;
                        }
          		return ret;
		}

    }
}
