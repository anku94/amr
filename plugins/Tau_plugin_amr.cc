/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include "amr_tracer.h"

#include <Profile/Profiler.h>
#include <Profile/TauAPI.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauSampling.h>
#include <Profile/TauTrace.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

int stop_tracing = 0;

tau::AMRTracer* tracer = nullptr;

/* lifecycle begin */
int Tau_plugin_event_post_init(Tau_plugin_event_post_init_data_t* data) {
  tracer = new tau::AMRTracer();
  return 0;
}

int Tau_plugin_event_pre_end_of_execution(
    Tau_plugin_event_pre_end_of_execution_data_t* data) {
  tracer->PrintStats();
  delete tracer;
  tracer = nullptr;
  return 0;
}

int Tau_plugin_event_end_of_execution(
    Tau_plugin_event_end_of_execution_data_t* data) {
  return 0;
}
/* lifecycle end */

/* event begin */
int Tau_plugin_event_function_entry(
    Tau_plugin_event_function_entry_data_t* data) {
  if (stop_tracing) return 0;

  if (!strncmp(data->timer_group, "TAU_USER", 8) == 0) return 0;

  tracer->MarkBegin(data->timer_name, data->timestamp);

  if (tracer->MyRank() == 0) {
    logf(LOG_DBUG, "FUNC ENTER %d: %s %s", tracer->MyRank(),
            data->timer_name, data->timer_group);
  }

  return 0;
}

int Tau_plugin_event_function_exit(
    Tau_plugin_event_function_exit_data_t* data) {
  if (stop_tracing) return 0;

  if (!strncmp(data->timer_group, "TAU_USER", 8) == 0) return 0;
  tracer->MarkEnd(data->timer_name, data->timestamp);

  if (tracer->MyRank() == 0) {
    logf(LOG_DBUG, "FUNC EXIT %d: %s %s", tracer->MyRank(), data->timer_name,
            data->timer_group);
  }

  return 0;
}

int Tau_plugin_event_send(Tau_plugin_event_send_data_t* data) {
  tracer->RegisterSend(data->message_tag, data->destination, data->bytes_sent,
                       data->timestamp);
  return 0;
}

int Tau_plugin_event_recv(Tau_plugin_event_recv_data_t* data) {
  tracer->RegisterRecv(data->message_tag, data->source, data->bytes_received,
                       data->timestamp);
  return 0;
}

/* event end */
/* trigger begin */

int Tau_plugin_event_trigger(Tau_plugin_event_trigger_data_t* data) {
  tracer->ProcessTriggerMsg(data->data);
  return 0;
}
/* trigger end */

extern "C" int Tau_plugin_init_func(int argc, char** argv, int id) {
  Tau_plugin_callbacks* cb =
      (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->PostInit = Tau_plugin_event_post_init;
  cb->FunctionEntry = Tau_plugin_event_function_entry;
  cb->FunctionExit = Tau_plugin_event_function_exit;
  cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution;
  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

  cb->Send = Tau_plugin_event_send;
  cb->Recv = Tau_plugin_event_recv;

  cb->Trigger = Tau_plugin_event_trigger;

  // cb.MetadataRegistrationComplete =
  // Tau_plugin_metadata_registration_complete_func;
  /* Trace events */
  // cb.Send = Tau_plugin_adios2_send;
  // cb.Recv = Tau_plugin_adios2_recv;
  // cb.AtomicEventTrigger = Tau_plugin_adios2_atomic_trigger;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

