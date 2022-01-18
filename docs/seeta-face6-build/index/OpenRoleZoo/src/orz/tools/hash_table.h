//
// Created by kier on 2019/3/21.
//

#ifndef ORZ_HASH_TABLE_H
#define ORZ_HASH_TABLE_H

#include <string>

namespace orz {
    inline unsigned int ELFhash(const char *str) {
        unsigned int hash = 0;
        unsigned int x = 0;
        while (*str) {
            hash = (hash << 4) + *str;
            if ((x = hash & 0xf0000000) != 0) {
                hash ^= (x >> 24);
                hash &= ~x;
            }
            str++;
        }
        return (hash & 0x7fffffff);
    }

    template <typename resources>
    class hash_node {
    public:
        using hash_type = unsigned int;

        hash_node() = default;

        explicit hash_node(const std::string &key, const resources &value)
                : key(key), hash(ELFhash(key.c_str())), value(value) {}

        std::string key;
        hash_type hash = 0;
        int64_t next = -1;

        resources value;
    };

    template <typename resources>
    class hash_table {
    public:
        using self = hash_table;
        using node_type = orz::hash_node<resources>;

        using hash_type = unsigned int;
        using index_type = hash_type;

        explicit hash_table(unsigned int size) : m_nodes(size_t(size > 2 ? size : 2), nullptr) {}

        hash_table() : hash_table(2) {}

        ~hash_table() {
            for (auto node : m_nodes) {
                delete node;
            }
        }

        hash_table(const self &) = delete;

        self &operator==(const self &) = delete;

        node_type *insert(const std::string &key, const resources &value) {
            auto *found_node = this->find(key);
            if (found_node) {
                found_node->value = value;
                return found_node;
            }
            auto *new_node = new node_type(key, value);
            insert(new_node);
            return new_node;
        }

        // TODO: add erase method
        node_type *find(const std::string &key) {
            auto hash = ELFhash(key.c_str());
            auto index = index_type(hash % m_nodes.size());
            while (m_nodes[index] != nullptr) {
                auto node = m_nodes[index];
                if (hash == node->hash && std::strcmp(key.c_str(), node->key.c_str()) == 0) {
                    return m_nodes[index];
                }
                auto next_index = m_nodes[index]->next;
                if (next_index < 0) break;
                index = index_type(next_index);
            }
            return nullptr;
        }

        size_t size() const {
            return m_size;
        }

        const std::vector<node_type *> &nodes() const {
            return m_nodes;
        }

    private:
        void insert(node_type *node) {
            if (node == nullptr) return;

            if (m_size >= m_nodes.size() / 2) {
                rehash(m_nodes.size() * 2);
            }

            ++m_size;
            auto hash = node->hash;
            auto index = index_type(hash % m_nodes.size());
            auto first_aid_index = index;
            if (m_nodes[index] == nullptr) {
                m_nodes[index] = node;
                return;
            }
            do {
                auto anchor = m_nodes[index];
                auto anchor_hash = anchor->hash;
                auto anchor_index = index_type(anchor_hash % m_nodes.size());
                // if (anchor_hash == hash && anchor->key == key) {}
                if (anchor_index == first_aid_index) {
                    auto next_index = anchor->next;
                    if (next_index < 0) {
                        // a.1 insert at empty slot
                        node->next = -1;
                        anchor->next = insert_at_next_ready(node);
                        break;  // break when m_nodes[index]->next < 0
                    }
                    index = index_type(next_index);
                } else {
                    // b.1 find pre-next value
                    auto pre_next_index = index_type(anchor_hash % m_nodes.size());
                    if (m_nodes[pre_next_index] == nullptr) {
                        m_nodes[pre_next_index] = anchor;
                        m_nodes[index] = node;
                        break;
                    }
                    while (m_nodes[pre_next_index]->next >= 0 &&
                           m_nodes[pre_next_index]->next != index) {
                        pre_next_index = index_type(m_nodes[pre_next_index]->next);
                    }
                    m_nodes[pre_next_index]->next = update_conflict_node(anchor);
                    m_nodes[index] = node;
                    break;  // break when hash conflict
                }
            } while (m_nodes[index] != nullptr);    ///< always true
        }

        index_type update_conflict_node(node_type *node) {
            auto hash = node->hash;
            auto index = index_type(hash % m_nodes.size());
            if (m_nodes[index] == nullptr) {
                m_nodes[index] = node;
                return index;
            }
            return insert_at_next_ready(node);
        }

        index_type insert_at_next_ready(node_type *node) {
            while (m_nodes[m_next_ready] != nullptr) m_next_ready = (m_next_ready + 1) % m_nodes.size();
            auto index = index_type(m_next_ready);
            m_next_ready = (m_next_ready + 1) % m_nodes.size();
            m_nodes[index] = node;
            return index;
        }

        void rehash(size_t size) {
            if (size < m_nodes.size()) size = m_nodes.size();
            hash_table temp((unsigned int) (size));
            for (auto &node : m_nodes) {
                if (node == nullptr) continue;
                node->next = -1;
                temp.insert(node);
            }
            m_nodes = temp.m_nodes;
            m_next_ready = temp.m_next_ready;
            m_size = temp.m_size;
            temp.m_nodes.clear();
            temp.m_next_ready = 0;
            temp.m_size = 0;
        }

    private:
        std::vector<node_type *> m_nodes;
        size_t m_next_ready = 0;
        size_t m_size = 0;
    };

}

#endif //ORZ_HASH_TABLE_H
