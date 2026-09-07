import { createRouter, createWebHistory } from 'vue-router'
import NewReview from '../views/NewReview.vue'
import History from '../views/History.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'new', component: NewReview },
    { path: '/history', name: 'history', component: History },
  ],
})

export default router
