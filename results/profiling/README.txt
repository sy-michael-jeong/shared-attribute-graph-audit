Relation profiling behind Table 3, produced by profile_relations.py.

  relation_profiling.json   coverage / cardinality / top-10 dominance /
                            class MI / train-test overlap for every relation
                            field in the five datasets. MI is measured against
                            the binary task label; the ISCX-VPN entry also
                            records the entropy of that label.

Train-test overlap is measured under the conventional stratified random split,
as described in Section 4.1.

CTU-13 screening (Section 7.5): Zeek extraction over the botnet captures found
72 TLS sessions in ssl.log against 31,744 connections in conn.log, i.e. 0.2%.
