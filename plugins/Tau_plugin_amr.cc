/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

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

int Tau_plugin_event_end_of_execution(
    Tau_plugin_event_end_of_execution_data_t* data) {
  return 0;
}

int Tau_plugin_event_function_entry(
    Tau_plugin_event_function_entry_data_t* data) {
  if (stop_tracing) return 0;

  fprintf(stderr, "FUNC ENTER %u: %s %s\n", data->tid, data->timer_name,
          data->timer_group);

  return 0;
}

int Tau_plugin_event_post_init(Tau_plugin_event_post_init_data_t* data) {}

int Tau_plugin_event_function_exit(
    Tau_plugin_event_function_exit_data_t* data) {
  if (stop_tracing) return 0;

  fprintf(stderr, "FUNC EXIT %u: %s %s\n", data->tid, data->timer_name,
          data->timer_group);

  return 0;
}

extern "C" int Tau_plugin_init_func(int argc, char** argv, int id) {
  Tau_plugin_callbacks* cb =
      (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->PostInit = Tau_plugin_event_post_init;
  cb->FunctionEntry = Tau_plugin_event_function_entry;
  cb->FunctionExit = Tau_plugin_event_function_exit;
  cb->EndOfExecution = Tau_plugin_event_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

