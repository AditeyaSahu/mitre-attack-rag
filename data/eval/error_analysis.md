# Error Analysis and Results Discussion

Auto-generated from `results.json` and `generations.json`. Figures are in `reports/figures/`.

## Overall metric comparison

| Variant | Context Rel. | Answer Rel. | Faithfulness | Citation Grd. | Hallu. Boundary |
|---|---|---|---|---|---|
| naive | 0.756 | 0.636 | 0.952 | 0.933 | 1.000 |
| hybrid | 0.800 | 0.661 | 0.946 | 0.867 | 1.000 |
| agentic | 0.756 | 0.633 | 0.930 | 1.000 | 1.000 |

## Agentic behaviour

- Queries decomposed into multiple sub-queries: **8/20**
- Answers revised after failed self-verification: **5/20**

## Lowest-scoring records: Faithfulness

- **hybrid** / `eval_010` (procedures) — score 0.545
  - Query: What real-world examples are there of the Phishing technique being used?
  - Answer (truncated): There are real-world examples of the Phishing technique (T1566) being used by threat actors and malware. To defend against Phishing, consider the following steps:
* Monitor for spearphishing emails that distribute malicious payloads, as seen in AppleJeus (G1049) [9]
* Implement security controls to prevent initial compromise via spear phishing, as used by Axiom (G0001) [10] [11]
* Conduct regular 
- **agentic** / `eval_001` (definition) — score 0.600
  - Query: What is phishing in MITRE ATT&CK?
  - Answer (truncated): Phishing in MITRE ATT&CK refers to a technique where adversaries utilize servers to stage, launch, and execute operations, including watering hole operations, enabling Phishing operations, or facilitating Command and Control (T1566). 
To understand Phishing in the context of MITRE ATT&CK, consider the following:
* Phishing is a technique (T1566) that involves using servers for various tasks, inclu
- **naive** / `eval_005` (id_lookup) — score 0.667
  - Query: What is the MITRE ATT&CK ID for spearphishing attachment?
  - Answer (truncated): The MITRE ATT&CK IDs for spearphishing attachment are T1566.001 and T1598.002. 
To identify the correct ID for your specific use case, consider the following:
* If the spearphishing attachment is related to the parent technique Phishing (T1566), use T1566.001.
* If the spearphishing attachment is related to the parent technique Phishing for Information (T1598), use T1598.002.
The context does not 
- **agentic** / `eval_011` (tactic_goal) — score 0.667
  - Query: What is the goal of the Initial Access tactic?
  - Answer (truncated): The goal of the Initial Access tactic (TA0001) is for the adversary to get into your network. This is based on the information provided by the MITRE ATT&CK context, which states that the adversary is trying to get into your network (TA0001). 

The retrieved context does not provide additional details about the specific actions or techniques used by the adversary in the Initial Access tactic beyond
- **hybrid** / `eval_013` (comparative) — score 0.778
  - Query: Compare phishing and spearphishing attachment in MITRE ATT&CK.
  - Answer (truncated): Phishing (T1566) and Spearphishing Attachment (T1566.001) are related but distinct concepts in MITRE ATT&CK. Phishing (T1566) is a technique that has 4 sub-techniques, including Spearphishing Attachment (T1566.001). Spearphishing Attachment (T1566.001) is a specific variant of spearphishing that employs the use of malware attached to an email. 

To compare the two:
* Phishing (T1566) is a broader 

## Lowest-scoring records: Context Relevance

- **naive** / `eval_013` (comparative) — score 0.000
  - Query: Compare phishing and spearphishing attachment in MITRE ATT&CK.
  - Answer (truncated): Phishing (T1566) and spearphishing attachment are not directly compared in the provided context. However, we can compare the tactics associated with Phishing (T1566) and other related techniques. 

Phishing (T1566) is associated with the Initial Access (TA0001) tactic. In contrast, Phishing for Information (T1598) and Gather Victim Identity Information (T1589) are associated with the Reconnaissanc
- **agentic** / `eval_013` (comparative) — score 0.000
  - Query: Compare phishing and spearphishing attachment in MITRE ATT&CK.
  - Answer (truncated): Phishing (T1566) and spearphishing, specifically Internal Spearphishing (T1534), are associated with different MITRE ATT&CK Enterprise tactics. Phishing (T1566) supports the Initial Access (TA0001) tactic, while Internal Spearphishing (T1534) supports the Lateral Movement (TA0008) tactic.

The retrieved context does not provide information on how to compare and mitigate these techniques in terms o
- **naive** / `eval_006` (sub_technique_listing) — score 0.333
  - Query: What are the sub-techniques of Phishing in MITRE ATT&CK?
  - Answer (truncated): The sub-techniques of Phishing (T1566) in MITRE ATT&CK Enterprise are:
* Spearphishing Attachment (T1566.001)
* Spearphishing Link (T1566.002)
* Spearphishing via Service (T1566.003)
* Spearphishing Voice (T1566.004) 
(T1566, https://attack.mitre.org/techniques/T1566).
- **naive** / `eval_007` (mitigation_lookup) — score 0.333
  - Query: What is the MITRE ATT&CK mitigation M1031?
  - Answer (truncated): The MITRE ATT&CK mitigation M1031 is Network Intrusion Prevention. To implement this mitigation, consider the following steps:
* Use network intrusion detection and prevention systems that utilize network signatures to identify traffic for specific adversary malware (M1031).
* Apply this mitigation to techniques including T1071, T1566, T1102, T1557, T1602, T1105, T1132, T1001, and 17 more (M1031).
- **naive** / `eval_008` (tactic_mapping) — score 0.333
  - Query: Which MITRE ATT&CK tactics does Process Injection support?
  - Answer (truncated): Process Injection (T1055) supports the following MITRE ATT&CK tactics: 
* Stealth (TA0005)
* Privilege Escalation (TA0004)
These tactics are associated with T1055 as per the provided context. 
Citations: T1055, TA0005, TA0004.

## Lowest-scoring records: Citation Grounding

- **naive** / `eval_003` (detection) — score 0.000
  - Query: How can I detect PowerShell abuse on Windows endpoints?
  - Answer (truncated): To detect PowerShell abuse on Windows endpoints, you can monitor for specific behavior chains. However, the provided context does not contain sufficient information to answer this question directly, as it does not mention PowerShell abuse detection explicitly.

The context does provide information on detecting Exploitation for Client Execution (T1203) and defending against Windows Management Instr
- **hybrid** / `eval_003` (detection) — score 0.000
  - Query: How can I detect PowerShell abuse on Windows endpoints?
  - Answer (truncated): To detect PowerShell abuse on Windows endpoints, you can monitor for suspicious activity related to Windows Management Instrumentation (WMI) execution, as PowerShell is often used to interact with WMI. 

Here are the steps to take:
* Monitor for adversarial abuse of WMI to execute local or remote commands via WMIC, PowerShell, or COM API through a multi-event chain: process creation, command execu
- **hybrid** / `eval_015` (detection_tactic_level) — score 0.000
  - Query: What monitoring should I put in place to detect lateral movement?
  - Answer (truncated): To detect lateral movement, you should monitor for suspicious file transfers, anomalous use of remote access tools, and unusual internal network activity. 
Here are the specific monitoring approaches:
* For Lateral Tool Transfer (T1570), monitor for:
  + Suspicious file transfers over SMB or Admin$ shares (AN0516, DET0183)
  + Anomalous use of scp, rsync, curl, or third-party sync apps transferrin
- **naive** / `eval_001` (definition) — score 1.000
  - Query: What is phishing in MITRE ATT&CK?
  - Answer (truncated): Phishing (T1566) is a MITRE ATT&CK Enterprise technique. 
To understand phishing in the context of MITRE ATT&CK, consider the following:
* Phishing (T1566) is associated with the Initial Access (TA0001) tactic, as seen in its application to gain initial access to a system.
* While the context does not provide a detailed definition of phishing, it mentions that adversaries may utilize servers for p
- **naive** / `eval_002` (mitigation) — score 1.000
  - Query: How do I defend against credential dumping?
  - Answer (truncated): To defend against credential dumping, specifically OS Credential Dumping (T1003), consider the following mitigations:
* Manage the access control list for "Replicating Directory Changes All" and other permissions associated with domain controller replication using Active Directory Configuration (M1015).
* Enable Attack Surface Reduction (ASR) rules to secure LSASS and prevent credential stealing o
