import { createRouter, createWebHistory } from 'vue-router'
import NewReview from '../views/NewReview.vue'
import History from '../views/History.vue'
import LlmSettings from '../views/LlmSettings.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'new', component: NewReview },
    { path: '/history', name: 'history', component: History },
    { path: '/settings', name: 'settings', component: LlmSettings },
  ],
})

export default router
