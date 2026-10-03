# SaaS Outage Dispute: the sample's own arguments

Reference only. These are the human-written arguments that ship with the DecisionLayer sample. caseagent never reads this file: every filing it produces is written by the agent from the dispute, the contract and the exhibits.

## Claim (LogisticsCo LLC)

Failure of Exculpatory Clause: The “User Responsibility” section contains a material drafting error because it states that the user waives claims against “FiveStar, Inc. and its affiliates,” rather than Respondent, SolidEdge, Inc. Exculpatory language must clearly identify the party being released. Because this provision releases a different entity and contains no express release of SolidEdge, it does not waive Claimant’s claims against Respondent.

Invalid Limitation of Liability: Although the Terms of Use attempt to cap damages at $100, enforcing that cap would deprive Claimant of any meaningful remedy for Respondent’s complete failure to provide its core data-ingestion service for 36 continuous hours. Claimant therefore contends that the limitation is unconscionable, fails of its essential purpose, and should not be enforced.
Basis for the $50,000 Demand: The requested $50,000 represents direct, nonduplicative operational expenses caused by the outage, calculated as follows:

1. $18,000 in emergency manual-processing labor: 10 operations employees worked 30 additional hours each at a loaded labor rate of $60 per hour to process shipments manually while the automated ingestion system was unavailable.

2. $12,500 in internal data-recovery labor: 5 data engineers spent 20 hours each at a loaded labor rate of $125 per hour rebuilding databases, reconciling shipment records, and validating data after service was restored.

3. $7,500 in third-party recovery and computing costs: Claimant incurred external technical-support, data-restoration, and additional cloud-computing charges required to restore and verify the affected data.

4. $12,000 in incremental logistics expenses: Claimant incurred additional carrier, shipment-rerouting, and expedited-delivery charges to clear the backlog created by the outage and prevent further delivery failures.

These four categories total exactly $50,000. The expenses would not have been incurred but for Respondent’s outage and were a foreseeable result of interrupting a data-ingestion service used to operate automated shipping pipelines. Claimant mitigated its damages by switching to manual processing, promptly rebuilding the affected data, and expediting delayed shipments. The demand does not include lost profits, reputational harm, punitive damages, or other speculative losses.

## Answer (SolidEdge, Inc.)

Obvious Scrivener's Error: The inclusion of "FiveStar, Inc." is a manifest typographical error (a scrivener's error). It is clear from the context of the entire agreement, including the preamble and the "Legal Notice" section, that the parties' true objective intent was to limit the liability of SolidEdge, Inc. The court or arbitrator should reform the contract to reflect this obvious intent.

Enforceability of the Damage Cap: In B2B transactions, courts routinely enforce liability caps and disclaimers of consequential damages. Claimant accepted these terms upon accessing the site and using the service. The $100 liability cap is valid, clear, and unambiguous.

Service Level Context: The Terms of Use govern the "web site and the content" on an "as-is" basis. Commercial SaaS performance metrics are typically governed by separate Service Level Agreements (SLAs), not standard website terms.

## Reply (LogisticsCo LLC)

Scrivener's Error Defense Inapplicable: Equitable reformation for a scrivener's error requires mutual mistake. Claimant had no hand in drafting this contract; it is a contract of adhesion presented on a take-it-or-leave-it basis. Respondent cannot claim "mutual mistake" for its own sloppy copy-pasting.

Scope of the Terms: Respondent's claim that these terms only cover the informational website is contradicted by their own text. The "Limitation of Liability" section explicitly references "materials on the web site or content," which includes the proprietary data ingestion pipelines, APIs, and cloud dashboards accessed directly through the platform web console.

Total Breach: A liability cap cannot be used to insulate a party from the consequences of a total failure of performance that frustrates the entire commercial purpose of the transaction.

## Response (SolidEdge, Inc.)

Integration and Severability: The TOS contains a standard "Severability of Provisions" clause. If the specific waiver sentence naming FiveStar is found to be problematic, it should be severed or read in harmony with the rest of the agreement—specifically the "Limitation of Liability" section, which explicitly names SolidEdge and successfully establishes the $100 limit.

No Gross Negligence Documented: Claimant has alleged an operational outage, which is a standard operational risk in cloud computing. Claimant has failed to plead or prove facts rising to the level of gross negligence or intentional misconduct necessary to pierce a B2B liability limitation.

Affirmation of Terms: By continuously utilizing the platform API endpoints hosted on this domain to ingest data, Claimant repeatedly manifested assent to the terms as posted, including the explicit $100 financial exposure limit.
