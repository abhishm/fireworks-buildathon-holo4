#!/bin/bash
# Screen cheap Fireworks models on 30 dev tasks (5 per domain)
set -a; . ./.env; set +a
cd AutomationBench
run() { # model input_cost output_cost tag
  uv run auto-bench --model "$1" --base-url https://api.fireworks.ai/inference/v1 --api-key-var FIREWORKS_API_KEY \
    --domains sales,marketing,operations,support,finance,hr --tasks "sales.docusign_renewal_automation,sales.zoom_calendar_conflict,sales.priority_selection,sales.historical_context_pipeline_report,sales.zoom_meeting_followup,marketing.social_roi,marketing.social_scheduling,marketing.content_syndication,marketing.linkedin_speaker_outreach,marketing.event_registration_sync,operations.safety_incident_investigation_routing,operations.hazmat_shipping_compliance,operations.safety_certification_gap_analysis,operations.drive_slack_dock_schedule,operations.contractor_badge_expiration,support.helpscout_health_sync,support.zendesk_maintenance_notify,support.zoho_sf_enrichment,support.intercom_freshdesk_escalation,support.freshdesk_sla_monitoring,finance.late_fee_calculation,finance.bank_fee_analysis,finance.annual_budget_prep,finance.weekly_expense_summary,finance.xero_po_matching,hr.jira_it_onboarding,hr.tuition_reimbursement,hr.expense_report_audit,hr.data_migration_validation,hr.anniversary_recognition" --max-concurrent 15 \
    --input-cost "$2" --output-cost "$3" --export-json "../runs/screen_$4.json" > "../runs/screen_$4.log" 2>&1
  echo "done $4: $(grep -E 'Pass rate|Average score|Estimated cost' ../runs/screen_$4.log | tr '\n' ' ')"
}
run accounts/fireworks/models/deepseek-v4p1-flash 0.0000003 0.0000012 deepseek &
run accounts/fireworks/models/nemotron-3-ultra-nvfp4 0.0000006 0.0000024 nemoultra &
run accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b 0.00000005 0.0000002 nemolight &
run accounts/fireworks/models/glm-5p3-flash 0.00000015 0.0000005 glmflash &
wait
